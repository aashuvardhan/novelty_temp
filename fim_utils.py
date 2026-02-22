"""
FIM-Guided Dynamic LoRA Placement Utilities
=============================================
Computes the Diagonal Empirical Fisher Information Matrix (FIM) to identify
which layers of a trained model are most sensitive to poisoned/forget-target
data, enabling surgical LoRA adapter placement.

Key Functions:
    - compute_diagonal_fim: Computes per-weight diagonal FIM from a data loader.
    - compute_layer_sensitivity: Compares FIM on clean vs. poisoned data to
      produce layer-wise sensitivity scores and dynamic target modules + ranks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def compute_diagonal_fim(model, data_loader, device, max_batches=10):
    """
    Compute the Diagonal Empirical Fisher Information Matrix for a model.

    For each weight θ_i, the diagonal FIM is approximated as:
        F_diag(θ_i) ≈ (1/|D|) * Σ_{x∈D} (∂ log p(y|x, θ) / ∂θ_i)^2

    Args:
        model: The neural network model (should be in eval mode externally).
        data_loader: DataLoader providing (data, target) batches.
        device: torch device (cuda/cpu).
        max_batches: Maximum number of batches to use for FIM estimation.

    Returns:
        fim_dict: dict mapping parameter name → diagonal FIM tensor (same shape as param).
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()

    # Initialize FIM accumulator with zeros for each parameter
    fim_dict = {}
    for name, param in model.named_parameters():
        fim_dict[name] = torch.zeros_like(param, device=device)

    total_samples = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break

        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)

        # Process each sample individually for accurate FIM
        for i in range(batch_size):
            model.zero_grad()

            single_input = data[i:i+1]
            single_target = target[i:i+1]

            output = model(single_input)
            log_probs = F.log_softmax(output, dim=1)

            # Use the true label's log probability
            loss = -log_probs[0, single_target[0]]
            loss.backward()

            # Accumulate squared gradients (diagonal FIM)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fim_dict[name] += param.grad.data ** 2

            total_samples += 1

    # Normalize by number of samples
    if total_samples > 0:
        for name in fim_dict:
            fim_dict[name] /= total_samples

    return fim_dict


def _get_layer_name(param_name):
    """
    Extract the layer name from a full parameter name.
    E.g., 'conv1.weight' → 'conv1', 'fc3.bias' → 'fc3',
          'layer4.0.conv2.weight' → 'layer4.0.conv2'
          'model.layer4.0.conv2.weight' → 'model.layer4.0.conv2'
    """
    parts = param_name.split('.')
    # Remove the last part (weight/bias)
    if parts[-1] in ('weight', 'bias'):
        return '.'.join(parts[:-1])
    return param_name


def compute_layer_sensitivity(model, clean_loader, poison_loader, device,
                               alpha=1.0, percentile=70, max_batches=10,
                               r_min=4, r_max=32):
    """
    Compute layer-wise sensitivity scores using FIM contrast between clean
    and poisoned data, then determine which layers to target with LoRA.

    The sensitivity score for layer L is:
        S(L) = Σ_{w∈L} (F_poison(w) - α * F_clean(w))

    Layers with S(L) > τ (the given percentile threshold) are flagged for
    LoRA injection. The LoRA rank for each flagged layer is scaled
    proportionally to its sensitivity score.

    Args:
        model: The trained global model (frozen Phase 1 model).
        clean_loader: DataLoader with clean (non-poisoned) client data.
        poison_loader: DataLoader with poisoned (forget-target) client data.
        device: torch device.
        alpha: Scaling factor for clean FIM (default 1.0). Prevents penalizing
               layers that are universally important for all tasks.
        percentile: Percentile threshold for flagging layers (default 70).
                    Higher = more selective/surgical.
        max_batches: Number of batches to use for each FIM computation.
        r_min: Minimum LoRA rank for mildly infected layers.
        r_max: Maximum LoRA rank for heavily infected layers.

    Returns:
        target_modules: List of layer names where S(L) > τ.
        rank_map: Dict mapping each target layer to its dynamic rank.
        sensitivity_scores: Full dict of layer_name → S(L) for logging.
    """

    print("\n" + "=" * 60)
    print("FIM-GUIDED LAYER SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Step 1: Compute FIM on clean data
    print("Computing FIM on clean data...")
    fim_clean = compute_diagonal_fim(model, clean_loader, device, max_batches)

    # Step 2: Compute FIM on poisoned data
    print("Computing FIM on poisoned data...")
    fim_poison = compute_diagonal_fim(model, poison_loader, device, max_batches)

    # Step 3: Compute per-layer sensitivity scores
    # Group parameters by layer name
    layer_params = {}
    for name in fim_clean.keys():
        layer_name = _get_layer_name(name)
        if layer_name not in layer_params:
            layer_params[layer_name] = []
        layer_params[layer_name].append(name)

    sensitivity_scores = {}
    for layer_name, param_names in layer_params.items():
        score = 0.0
        for pname in param_names:
            if pname in fim_poison and pname in fim_clean:
                delta = fim_poison[pname] - alpha * fim_clean[pname]
                # Sum all positive deltas (we care about weights that fire MORE
                # for poisoned data than clean data)
                score += torch.clamp(delta, min=0).sum().item()
        sensitivity_scores[layer_name] = score

    # Step 4: Filter out layers that cannot have LoRA (e.g., BatchNorm, pooling)
    # LoRA can only be applied to nn.Linear and nn.Conv2d layers
    eligible_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if name in sensitivity_scores:
                eligible_layers[name] = sensitivity_scores[name]

    if len(eligible_layers) == 0:
        print("WARNING: No eligible layers found for LoRA injection!")
        print("Falling back to all Linear/Conv2d layers.")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                eligible_layers[name] = sensitivity_scores.get(name, 0.0)

    # Step 5: Compute threshold
    scores = list(eligible_layers.values())
    threshold = np.percentile(scores, percentile)

    # Step 6: Flag layers above threshold
    target_modules = []
    flagged_scores = {}
    for layer_name, score in eligible_layers.items():
        if score > threshold:
            target_modules.append(layer_name)
            flagged_scores[layer_name] = score

    # Safety: ensure at least one layer is selected
    if len(target_modules) == 0:
        print("WARNING: No layers exceeded threshold. Selecting top layer.")
        top_layer = max(eligible_layers, key=eligible_layers.get)
        target_modules = [top_layer]
        flagged_scores = {top_layer: eligible_layers[top_layer]}

    # Step 7: Compute dynamic ranks proportional to sensitivity
    max_score = max(flagged_scores.values()) if flagged_scores else 1.0
    rank_map = {}
    for layer_name, score in flagged_scores.items():
        if max_score > 0:
            normalized = score / max_score
        else:
            normalized = 0.5
        rank = int(r_min + normalized * (r_max - r_min))
        # Round to nearest power of 2 for efficiency
        rank = max(r_min, min(r_max, rank))
        rank_map[layer_name] = rank

    # Print diagnostic report
    print("\n--- Layer Sensitivity Report ---")
    print(f"{'Layer':<40} {'Score':>12} {'Status':>10} {'Rank':>6}")
    print("-" * 72)
    for layer_name in sorted(eligible_layers.keys()):
        score = eligible_layers[layer_name]
        if layer_name in target_modules:
            status = "FLAGGED"
            rank = rank_map[layer_name]
            print(f"{layer_name:<40} {score:>12.4f} {status:>10} r={rank:>4}")
        else:
            print(f"{layer_name:<40} {score:>12.4f} {'clean':>10} {'--':>6}")

    print(f"\nThreshold (τ at {percentile}th percentile): {threshold:.4f}")
    print(f"Total eligible layers: {len(eligible_layers)}")
    print(f"Flagged layers: {len(target_modules)}")
    print(f"Target modules: {target_modules}")
    print(f"Rank map: {rank_map}")
    print("=" * 60 + "\n")

    return target_modules, rank_map, sensitivity_scores
