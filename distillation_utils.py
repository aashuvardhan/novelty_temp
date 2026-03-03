"""
Feature Distribution Matching (DM) — Dataset Distillation Utilities
====================================================================
Compresses a client's real dataset into a tiny set of synthetic images
whose deep feature embeddings match the real data's distribution.

The synthetic images are optimized so that their Mean Feature Vector
(extracted by a frozen model) matches the Mean Feature Vector of the
real images.  This is measured via Maximum Mean Discrepancy (MMD).

Key Components:
    - FeatureExtractor: Wraps a model, strips the final FC, returns embeddings.
    - distribution_matching_loss: Computes MMD between real and synthetic embeddings.
    - distill_client_data: The main optimisation loop (optimises pixels, not weights).

Usage:
    Called once *after* Phase 1 federated training (Phase 1.5), using the
    fully trained global model so that the feature extractor produces
    meaningful embeddings.  The resulting tensors are saved as .pt files
    and loaded during Phase 2 unlearning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy


# ======================================================================
# 1. Feature Extractor — works for both LeNet and ResNet18
# ======================================================================
class FeatureExtractor(nn.Module):
    """
    Strips the final classification layer of a model and returns embeddings.

    Supports:
        - LeNet_FashionMNIST  (custom sequential: conv1→conv2→fc1→fc2→fc3)
        - CNN_Cifar10 ResNet18 wrapper  (self.model = torchvision.resnet18)
    """

    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.model_type = self._detect_model_type()
        self._prepare()

    def _detect_model_type(self):
        """Detect if this is a ResNet18 wrapper or a LeNet-style model."""
        if hasattr(self.base_model, 'model'):
            # CNN_Cifar10 wraps torchvision ResNet18 inside self.model
            return 'resnet18'
        elif hasattr(self.base_model, 'fc3'):
            # LeNet_FashionMNIST has fc1, fc2, fc3
            return 'lenet'
        else:
            # Fallback: try nn.Sequential minus last child
            return 'generic'

    def _prepare(self):
        """Neutralise the final FC layer so forward returns embeddings."""
        if self.model_type == 'resnet18':
            # Replace ResNet fc with Identity → forward returns 512-d embeddings
            self.base_model.model.fc = nn.Identity()
        elif self.model_type == 'lenet':
            # Replace fc3 with Identity → forward returns 84-d embeddings
            self.base_model.fc3 = nn.Identity()
        else:
            # Generic: strip last child
            children = list(self.base_model.children())
            self.base_model = nn.Sequential(*children[:-1])

    def forward(self, x):
        out = self.base_model(x)
        # Flatten in case of spatial dims left over (e.g. ResNet avgpool)
        return out.view(out.size(0), -1)


# ======================================================================
# 2. MMD Loss
# ======================================================================
def distribution_matching_loss(real_features, syn_features):
    """
    Maximum Mean Discrepancy between the mean embeddings of real and
    synthetic data.

    L = || mean(Φ(x_real)) - mean(Φ(x_syn)) ||^2

    Args:
        real_features: (N_real, D)  — detached, no grad needed
        syn_features:  (N_syn, D)   — requires grad (flows to pixels)

    Returns:
        Scalar loss tensor.
    """
    mean_real = torch.mean(real_features, dim=0)
    mean_syn  = torch.mean(syn_features,  dim=0)
    return torch.sum((mean_real - mean_syn) ** 2)


# ======================================================================
# 3. Helper: collect all images per class from a DataLoader
# ======================================================================
def _collect_images_by_class(data_loader, num_classes, device):
    """
    Iterate through a DataLoader and bucket images by class label.

    Returns:
        dict {class_id: Tensor of shape (N_c, C, H, W)}
    """
    buckets = {c: [] for c in range(num_classes)}
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        for c in range(num_classes):
            mask = (target == c)
            if mask.any():
                buckets[c].append(data[mask])
    # Concatenate per class
    result = {}
    for c in range(num_classes):
        if len(buckets[c]) > 0:
            result[c] = torch.cat(buckets[c], dim=0)
    return result


# ======================================================================
# 4. Core Distillation Loop
# ======================================================================
@torch.enable_grad()
def distill_client_data(real_loader, base_model, num_classes=10,
                        ipc=5, num_channels=1, img_size=28,
                        dm_iterations=500, lr=0.01, device='cuda'):
    """
    Run Feature Distribution Matching on one client's data.

    Optimises *pixel values* of synthetic images until their mean
    feature embedding matches the real data's, class-by-class.

    Args:
        real_loader:   DataLoader for this client's local data.
        base_model:    The global model (frozen), used as feature extractor.
        num_classes:   Number of classes in the dataset.
        ipc:           Images Per Class to synthesise.
        num_channels:  1 for FashionMNIST, 3 for CIFAR-10.
        img_size:      28 for FashionMNIST, 32 for CIFAR-10.
        dm_iterations: Number of pixel-optimisation iterations.
        lr:            Learning rate for the pixel optimiser.
        device:        'cuda' or 'cpu'.

    Returns:
        (syn_images, syn_labels)  — both detached Tensors on CPU.
    """
    # --- Set up frozen feature extractor ---
    extractor = FeatureExtractor(base_model).to(device)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    # --- Collect real images bucketed by class ---
    class_images = _collect_images_by_class(real_loader, num_classes, device)
    present_classes = sorted(class_images.keys())

    if len(present_classes) == 0:
        # Edge case: empty client (shouldn't normally happen)
        empty_imgs = torch.zeros(0, num_channels, img_size, img_size)
        empty_lbls = torch.zeros(0, dtype=torch.long)
        return empty_imgs, empty_lbls

    # --- Pre-compute real feature means (constant, no grad) ---
    real_means = {}
    with torch.no_grad():
        for c in present_classes:
            real_feat = extractor(class_images[c])
            real_means[c] = torch.mean(real_feat, dim=0)  # (D,)

    # --- Initialise synthetic images as learnable parameters ---
    syn_per_class = {}
    for c in present_classes:
        syn_imgs = torch.randn(ipc, num_channels, img_size, img_size,
                               device=device, requires_grad=True)
        syn_per_class[c] = syn_imgs

    # All synthetic pixel tensors go into one optimiser
    optimizer_img = optim.Adam(list(syn_per_class.values()), lr=lr)

    # --- Optimisation loop ---
    for it in range(dm_iterations):
        optimizer_img.zero_grad()
        total_loss = 0.0

        for c in present_classes:
            # Clamp synthetic pixels to [0, 1] via sigmoid to prevent
            # pixel explosion and ensure valid input to the extractor
            valid_syn = torch.sigmoid(syn_per_class[c])
            syn_feat = extractor(valid_syn)
            mean_syn = torch.mean(syn_feat, dim=0)
            loss_c = torch.sum((real_means[c] - mean_syn) ** 2)
            total_loss = total_loss + loss_c

        total_loss.backward()
        optimizer_img.step()

        if (it + 1) % 100 == 0:
            print(f"  [DM] Iteration {it+1}/{dm_iterations}, Loss: {total_loss.item():.6f}")

    # --- Assemble final (images, labels) tensors ---
    all_imgs = []
    all_lbls = []
    for c in present_classes:
        # Save the sigmoid-clamped version so pixels are in [0, 1]
        all_imgs.append(torch.sigmoid(syn_per_class[c]).detach().cpu())
        all_lbls.append(torch.full((ipc,), c, dtype=torch.long))

    syn_images = torch.cat(all_imgs, dim=0)
    syn_labels = torch.cat(all_lbls, dim=0)
    return syn_images, syn_labels


# ======================================================================
# 5. Top-level function: distill ALL clients and save to folder
# ======================================================================
def distill_all_clients(client_all_loaders, base_model, args,
                        save_dir='distilled_data'):
    """
    Pre-compute distilled datasets for every client and persist them.

    Args:
        client_all_loaders: list of DataLoaders, one per client.
        base_model:         The fully trained global model (after Phase 1 FedAvg).
        args:               Parsed CLI args (num_user, num_classes, data_name,
                            device, ipc, dm_iterations).
        save_dir:           Folder to write .pt files into.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Infer image properties from dataset
    if args.data_name in ('fashionmnist', 'mnist'):
        num_channels, img_size = 1, 28
    elif args.data_name == 'cifar10':
        num_channels, img_size = 3, 32
    elif args.data_name == 'cifar100':
        num_channels, img_size = 3, 32
    else:
        num_channels, img_size = 1, 28  # safe default

    ipc = getattr(args, 'ipc', 5)
    dm_iterations = getattr(args, 'dm_iterations', 500)

    print("=" * 60)
    print("PHASE 1.5 — FEATURE DISTRIBUTION MATCHING (DATASET DISTILLATION)")
    print(f"  Clients: {args.num_user}")
    print(f"  Images Per Class (IPC): {ipc}")
    print(f"  Image Shape: ({num_channels}, {img_size}, {img_size})")
    print(f"  Optimisation Iterations: {dm_iterations}")
    print(f"  Save directory: {save_dir}/")
    print("=" * 60)

    for client_id in range(args.num_user):
        print(f"\n[Client {client_id}] Distilling local data ...")
        local_loader = client_all_loaders[client_id]

        syn_images, syn_labels = distill_client_data(
            real_loader=local_loader,
            base_model=base_model,
            num_classes=args.num_classes,
            ipc=ipc,
            num_channels=num_channels,
            img_size=img_size,
            dm_iterations=dm_iterations,
            device=str(args.device),
        )

        save_path = os.path.join(save_dir, f'client_{client_id}.pt')
        torch.save((syn_images, syn_labels), save_path)
        print(f"  → Saved {syn_images.shape[0]} synthetic images to {save_path}")

    print("\n" + "=" * 60)
    print("PHASE 0 COMPLETE — All clients distilled successfully.")
    print("=" * 60 + "\n")
