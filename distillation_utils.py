"""
Feature Distribution Matching (DM) — Dataset Distillation Utilities
====================================================================
Compresses a client's real dataset into a tiny set of synthetic images
whose deep feature embeddings match the real data's distribution.

The synthetic images are optimised so that their Mean Feature Vector
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

Pixel Normalisation:
    Real images loaded through the data pipeline are already normalised by
    the dataset-specific mean/std.  Synthetic pixels MUST be initialised and
    clamped in exactly the same normalised space so that the feature extractor
    sees the same pixel-value range as it does for real images.  For each
    dataset we therefore pass pixel_mean / pixel_std and derive per-channel
    clamp bounds as:
        norm_min[c] = (0 - mean[c]) / std[c]
        norm_max[c] = (1 - mean[c]) / std[c]
    This matches the ToTensor() + Normalize() pipeline used in data_utils.py.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np


# ======================================================================
# Dataset-specific normalisation constants
# (must match the transforms in dataset/data_utils.py)
# ======================================================================
DATASET_STATS = {
    'fashionmnist': {
        'mean': np.array([0.1307]),
        'std':  np.array([0.3081]),
    },
    'mnist': {
        'mean': np.array([0.1307]),
        'std':  np.array([0.3081]),
    },
    'cifar10': {
        'mean': np.array([0.4914, 0.4822, 0.4465]),
        'std':  np.array([0.2023, 0.1994, 0.2010]),
    },
    'cifar100': {
        'mean': np.array([0.4914, 0.4822, 0.4465]),
        'std':  np.array([0.2023, 0.1994, 0.2010]),
    },
}


# ======================================================================
# 1. Feature Extractor — works for both LeNet and ResNet18
# ======================================================================
class FeatureExtractor(nn.Module):
    """
    Strips the final classification layer of a model and returns embeddings.

    Supports:
        - LeNet_FashionMNIST  (custom sequential: conv1→conv2→fc1→fc2→fc3)
        - CNN_Cifar10 / CNN_Cifar100 ResNet18 wrapper  (self.model = torchvision.resnet18)
    """

    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.model_type = self._detect_model_type()
        self._prepare()

    def _detect_model_type(self):
        """Detect if this is a ResNet18 wrapper or a LeNet-style model."""
        if hasattr(self.base_model, 'model'):
            # CNN_Cifar10 / CNN_Cifar100 wrap torchvision ResNet18 inside self.model
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

    Features are L2-normalised before computing means so that the loss
    is bounded and comparable across different feature magnitudes.

    Args:
        real_features: (N_real, D)  — detached, no grad needed
        syn_features:  (N_syn, D)   — requires grad (flows to pixels)

    Returns:
        Scalar loss tensor.
    """
    real_features = F.normalize(real_features, p=2, dim=1)
    syn_features  = F.normalize(syn_features,  p=2, dim=1)
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
                        dm_iterations=500, lr=0.1, device='cuda',
                        pixel_mean=None, pixel_std=None):
    """
    Run Feature Distribution Matching on one client's data.

    Optimises *pixel values* of synthetic images until their mean
    feature embedding matches the real data's, class-by-class.

    The synthetic pixels are initialised and clamped in the same
    normalised pixel space as the real images (using pixel_mean /
    pixel_std to derive per-channel bounds), so the feature extractor
    sees the same input distribution for both real and synthetic data.

    Args:
        real_loader:   DataLoader for this client's local data.
        base_model:    The global model (frozen), used as feature extractor.
        num_classes:   Number of classes in the dataset.
        ipc:           Images Per Class to synthesise.
        num_channels:  1 for FashionMNIST, 3 for CIFAR-10/100.
        img_size:      28 for FashionMNIST, 32 for CIFAR-10/100.
        dm_iterations: Number of pixel-optimisation iterations.
        lr:            Learning rate for the pixel optimiser.
        device:        'cuda' or 'cpu'.
        pixel_mean:    Per-channel normalisation mean (numpy array, length C).
                       Defaults to zeros (no-op) if None.
        pixel_std:     Per-channel normalisation std  (numpy array, length C).
                       Defaults to ones (no-op) if None.

    Returns:
        (syn_images, syn_labels)  — both detached Tensors on CPU.
    """
    # --- Pixel clamp bounds in normalised space ---
    # Real images from the DataLoader live in [(0-mean)/std, (1-mean)/std]
    # per channel; synthetic pixels must stay in the same range.
    if pixel_mean is None:
        pixel_mean = np.zeros(num_channels)
    if pixel_std is None:
        pixel_std = np.ones(num_channels)

    # norm_min/max: shape (C, 1, 1) tensors for broadcasting
    norm_min = torch.tensor(
        (0.0 - pixel_mean) / pixel_std, dtype=torch.float32
    ).view(num_channels, 1, 1).to(device)
    norm_max = torch.tensor(
        (1.0 - pixel_mean) / pixel_std, dtype=torch.float32
    ).view(num_channels, 1, 1).to(device)

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
    # L2-normalise embeddings before computing means so the MMD loss
    # is bounded and converges reliably.
    real_means = {}
    with torch.no_grad():
        for c in present_classes:
            real_feat = extractor(class_images[c])
            real_feat = F.normalize(real_feat, p=2, dim=1)
            real_means[c] = torch.mean(real_feat, dim=0)  # (D,)

    # --- Initialise synthetic images as learnable parameters ---
    # Initialise in normalised pixel space so the extractor starts from a
    # plausible input range (same as real images, not stuck in [0, 1]).
    syn_per_class = {}
    for c in present_classes:
        # Start with small random values around zero in normalised space
        syn_imgs = torch.randn(ipc, num_channels, img_size, img_size,
                               device=device) * 0.1
        # Clamp to valid normalised range immediately
        syn_imgs = torch.max(torch.min(syn_imgs, norm_max), norm_min)
        syn_imgs = syn_imgs.detach().requires_grad_(True)
        syn_per_class[c] = syn_imgs

    # All synthetic pixel tensors go into one optimiser
    optimizer_img = optim.Adam(list(syn_per_class.values()), lr=lr)

    # --- Optimisation loop ---
    for it in range(dm_iterations):
        optimizer_img.zero_grad()
        total_loss = 0.0

        for c in present_classes:
            # Feed synthetic pixels directly (no sigmoid — they are already
            # in the normalised space that the feature extractor expects)
            syn_feat = extractor(syn_per_class[c])
            syn_feat = F.normalize(syn_feat, p=2, dim=1)
            mean_syn = torch.mean(syn_feat, dim=0)
            loss_c = torch.sum((real_means[c] - mean_syn) ** 2)
            total_loss = total_loss + loss_c

        # Average across classes so loss is comparable regardless of
        # how many classes a client has
        total_loss = total_loss / len(present_classes)

        total_loss.backward()
        optimizer_img.step()

        # Clamp each class tensor back to the valid normalised pixel range
        with torch.no_grad():
            for c in present_classes:
                syn_per_class[c].data = torch.max(
                    torch.min(syn_per_class[c].data, norm_max), norm_min
                )

        if (it + 1) % 100 == 0:
            print(f"  [DM] Iteration {it+1}/{dm_iterations}, Loss: {total_loss.item():.6f}")

    # --- Assemble final (images, labels) tensors ---
    all_imgs = []
    all_lbls = []
    for c in present_classes:
        # Save the clamped normalised pixels (already in correct range)
        all_imgs.append(syn_per_class[c].detach().cpu())
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
    elif args.data_name in ('cifar10', 'cifar100'):
        num_channels, img_size = 3, 32
    else:
        num_channels, img_size = 1, 28  # safe default

    # Per-dataset normalisation constants (must match data_utils.py transforms)
    stats = DATASET_STATS.get(args.data_name, None)
    if stats is not None:
        pixel_mean = stats['mean']
        pixel_std  = stats['std']
    else:
        pixel_mean = np.zeros(num_channels)
        pixel_std  = np.ones(num_channels)

    ipc = getattr(args, 'ipc', 5)
    dm_iterations = getattr(args, 'dm_iterations', 500)

    print("=" * 60)
    print("PHASE 1.5 — FEATURE DISTRIBUTION MATCHING (DATASET DISTILLATION)")
    print(f"  Dataset: {args.data_name}")
    print(f"  Clients: {args.num_user}")
    print(f"  Images Per Class (IPC): {ipc}")
    print(f"  Image Shape: ({num_channels}, {img_size}, {img_size})")
    print(f"  Pixel mean: {pixel_mean}  std: {pixel_std}")
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
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )

        save_path = os.path.join(save_dir, f'client_{client_id}.pt')
        torch.save((syn_images, syn_labels), save_path)
        print(f"  → Saved {syn_images.shape[0]} synthetic images to {save_path}")

    print("\n" + "=" * 60)
    print("PHASE 0 COMPLETE — All clients distilled successfully.")
    print("=" * 60 + "\n")
