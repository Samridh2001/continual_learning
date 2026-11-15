import torch
import numpy as np

def compute_ece(logits, targets, n_bins=15):
    """
    Computes Expected Calibration Error (ECE).
    Args:
        logits (torch.Tensor or np.ndarray): [N, C] unnormalized model logits or probabilities
        targets (torch.Tensor or np.ndarray): [N] ground-truth class indices
        n_bins (int): Number of confidence bins (standard is 15)
    Returns:
        ece (float): Expected calibration error percentage (0 to 100)
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    softmaxes = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece.item() * 100.0)


def compute_grassmann_angles(basis1, basis2):
    """
    Computes Grassmann principal angles between two orthonormal subspaces.
    Args:
        basis1 (np.ndarray): [D, r1]
        basis2 (np.ndarray): [D, r2]
    Returns:
        mean_cos (float): Mean cosine of principal angles (1.0 = identical, 0.0 = orthogonal)
        min_angle_deg (float): Minimum principal angle in degrees
    """
    # Orthonormalize if needed
    Q1, _ = np.linalg.qr(basis1)
    Q2, _ = np.linalg.qr(basis2)

    # SVD of Q1^T @ Q2 gives cosines of principal angles
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)

    angles_rad = np.arccos(s)
    angles_deg = np.degrees(angles_rad)

    return float(np.mean(s)), float(np.min(angles_deg)) if len(angles_deg) > 0 else 90.0
