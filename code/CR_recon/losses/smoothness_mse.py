"""
SmoothnessMSELoss: MSE + Total Variation (TV) regularization
Encourages smooth spectral predictions by penalizing adjacent bin differences.

Formula: L = mse_weight * MSE(pred, target) + tv_weight * TV(pred)
"""
import torch
import torch.nn.functional as F


def total_variation(x):
    """
    Compute Total Variation along the spectrum dimension.

    TV measures the sum of absolute differences between adjacent elements.
    Lower TV = smoother prediction.

    Args:
        x: (B, C, L) where C=4, L=out_len

    Returns:
        tv_loss: Scalar
    """
    # Differences between adjacent bins along the last dimension
    diff = x[:, :, 1:] - x[:, :, :-1]

    # L1 norm (absolute differences)
    tv = torch.abs(diff).mean()

    return tv


def get_smoothness_mse_loss(mse_weight=0.7, tv_weight=0.3):
    """
    Factory function for MSE + Total Variation loss.

    Combines MSE loss with Total Variation (TV) regularization.

    Total Variation penalizes large differences between adjacent spectrum bins,
    encouraging smooth predictions that are physically meaningful.

    Args:
        mse_weight: Weight of MSE loss (0 to 1)
        tv_weight: Weight of TV regularization (0 to 1)

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

        # MSE loss
        mse_loss = F.mse_loss(pred, tgt)

        # Total Variation loss (smoothness regularization)
        tv_loss = total_variation(pred)

        # Combine
        return mse_weight * mse_loss + tv_weight * tv_loss

    return loss_fn
