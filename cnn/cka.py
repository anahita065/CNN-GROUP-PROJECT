"""Linear Centered Kernel Alignment (CKA) and feature extraction utilities."""

import torch
import numpy as np
from typing import Dict, List, Tuple


# ------------------------------------------------------------------
# Linear CKA
# ------------------------------------------------------------------

def _center(X: torch.Tensor) -> torch.Tensor:
    """Mean-center columns of X."""
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X: (n_samples, d_x) — centred or uncentred features.
        Y: (n_samples, d_y)

    Returns:
        Scalar CKA value in [0, 1].
    """
    X = _center(X.float())
    Y = _center(Y.float())

    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    hsic_xy = (X.T @ Y).norm(p="fro").pow(2)
    hsic_xx = (X.T @ X).norm(p="fro")
    hsic_yy = (Y.T @ Y).norm(p="fro")

    denom = hsic_xx * hsic_yy
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


def compute_cka_matrix(features: Dict[str, torch.Tensor]) -> np.ndarray:
    """Return a symmetric pairwise CKA matrix over all collected layers."""
    names = list(features.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i <= j:
                val = linear_cka(features[ni], features[nj])
                mat[i, j] = mat[j, i] = val
    return mat


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model,
    loader,
    layer_names: List[str],
    num_samples: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run the model on `num_samples` validation images and collect layer activations.

    Spatial dimensions are already averaged inside the model's hook, so each
    layer contributes a (n_samples, C) tensor — ready for CKA.
    """
    model.eval()
    model.register_hooks(layer_names)

    buckets: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
    collected = 0

    for images, _ in loader:
        if collected >= num_samples:
            break
        model(images.to(device))
        for name, feat in model.get_features().items():
            buckets[name].append(feat.cpu())
        collected += images.size(0)

    model.remove_hooks()

    result: Dict[str, torch.Tensor] = {}
    for name in layer_names:
        if buckets[name]:
            result[name] = torch.cat(buckets[name], dim=0)[:num_samples].float()
    return result


@torch.no_grad()
def extract_features_with_labels(
    model,
    loader,
    layer_name: str,
    num_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (features, labels) for a single layer — used for t-SNE."""
    model.eval()
    model.register_hooks([layer_name])

    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    collected = 0

    for images, labels in loader:
        if collected >= num_samples:
            break
        model(images.to(device))
        feat = model.get_features().get(layer_name)
        if feat is not None:
            all_feats.append(feat.cpu())
            all_labels.append(labels)
        collected += images.size(0)

    model.remove_hooks()

    feats  = torch.cat(all_feats,  dim=0)[:num_samples].float()
    labels = torch.cat(all_labels, dim=0)[:num_samples]
    return feats, labels
