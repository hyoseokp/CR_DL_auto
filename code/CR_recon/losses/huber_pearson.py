"""
HuberPearsonLoss: Huber loss + Pearson correlation
Robust to outliers while maintaining spectral shape correlation.

Formula: L = (1 - pearson_weight) * Huber(pred, target) + pearson_weight * (1 - Pearson)
"""
import torch
import torch.nn.functional as F


def pearson_loss(pred, target, eps=1e-8):
    """
    Compute Pearson correlation loss.

    Pearson correlation measures the linear relationship between prediction and target.
    Loss = 1 - mean(Pearson correlation)

    Args:
        pred: (B, C, L) where C=4, L=out_len
        target: Same shape as pred
        eps: Small value for numerical stability

    Returns:
        loss: Scalar
    """
    # Center the values
    pred_mean = pred.mean(dim=-1, keepdim=True)
    target_mean = target.mean(dim=-1, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Compute covariance and standard deviations
    cov = (pred_centered * target_centered).mean(dim=-1)
    std_pred = pred_centered.std(dim=-1)
    std_target = target_centered.std(dim=-1)

    # Pearson correlation coefficient
    pearson = cov / (std_pred * std_target + eps)

    # Clamp to [-1, 1] to avoid NaN
    pearson = torch.clamp(pearson, -1.0, 1.0)

    # Loss: 1 - correlation (higher correlation = lower loss)
    loss = (1.0 - pearson).mean()

    return loss


def get_huber_pearson_loss(huber_delta=0.5, pearson_weight=0.2):
    """
    Factory function for Huber + Pearson Correlation loss.

    Combines Huber loss (robust to outliers) with Pearson correlation loss.

    Args:
        huber_delta: Delta parameter for Huber loss. Smaller values = more robust to outliers
        pearson_weight: Weight of Pearson correlation loss (0 to 1)

    Returns:
        callable: loss_fn(pred, tgt) -> scalar
    """
    def loss_fn(pred, tgt):
        pred = pred.float()
        tgt = tgt.float()

        # Normalize to (B, 4, out_len) if needed
        if pred.dim() == 4:  # (B, 2, 2, out_len)
            pred = pred.view(pred.shape[0], -1, pred.shape[-1])
            tgt = tgt.view(tgt.shape[0], -1, tgt.shape[-1])

        # Huber loss
        huber_loss = F.smooth_l1_loss(pred, tgt, beta=huber_delta)

        # Pearson correlation loss
        pearson_loss_val = pearson_loss(pred, tgt)

        # Combine
        return (1.0 - pearson_weight) * huber_loss + pearson_weight * pearson_loss_val

    return loss_fn
