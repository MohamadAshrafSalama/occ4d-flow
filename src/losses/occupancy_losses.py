from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """
    Alpha-balanced focal loss for binary occupancy prediction.

    Focal loss down-weights the contribution of easy (well-classified)
    examples so that training focuses on hard negatives/positives.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1, else p_t = 1-p.

    alpha controls the balance between positive and negative classes.
    gamma=2 is the standard value from the original paper.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, X, Y, Z) or (B, X, Y, Z) raw predictions
            targets: same shape as logits, binary labels in {0, 1}
            weight: optional voxel-wise weight tensor, same shape

        Returns:
            loss: scalar
        """
        if logits.dim() == 5 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        targets = targets.float()
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if weight is not None:
            loss = loss * weight

        return loss.mean()


class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax surrogate loss for maximizing IoU.

    The Lovasz extension provides a convex surrogate for the
    discrete IoU loss. For binary occupancy:
        - Compute error vector: e_i = 1 - p_i if y_i=1, else e_i = p_i
        - Sort errors in decreasing order
        - Apply Lovasz extension (piecewise linear interpolation of IoU)
        - Return mean over batch

    This implementation handles the binary case specifically.
    """

    def __init__(self, per_image: bool = True) -> None:
        super().__init__()
        self.per_image = per_image

    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovasz extension (binary case)."""
        n = gt_sorted.shape[0]
        gts = gt_sorted.sum()
        inter = gts - gt_sorted.cumsum(0)
        union = gts + (1.0 - gt_sorted).cumsum(0)
        iou = 1.0 - inter / union.clamp(min=1e-10)
        if n > 1:
            iou[1:] = iou[1:] - iou[:-1]
        return iou

    def _lovasz_binary(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.float()
        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = (F.relu(errors_sorted) * grad).sum()
        return loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, X, Y, Z) or (B, X, Y, Z)
            targets: same shape, binary

        Returns:
            loss: scalar
        """
        if logits.dim() == 5 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        B = logits.shape[0]

        if self.per_image:
            losses = []
            for b in range(B):
                log_b = logits[b].reshape(-1)
                tgt_b = targets[b].reshape(-1).float()
                losses.append(self._lovasz_binary(log_b, tgt_b))
            return torch.stack(losses).mean()
        else:
            return self._lovasz_binary(logits.reshape(-1), targets.reshape(-1).float())


class DiceLoss(nn.Module):
    """
    Soft Dice loss for volumetric binary segmentation.

        Dice = 2 * |A ∩ B| / (|A| + |B|)
        Loss = 1 - Dice

    Operates on predicted probabilities (after sigmoid) and targets.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        prob: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prob: (B, 1, X, Y, Z) or (B, X, Y, Z) predicted probability
            targets: same shape, binary labels

        Returns:
            loss: scalar
        """
        if prob.dim() == 5 and prob.shape[1] == 1:
            prob = prob.squeeze(1)
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        targets = targets.float()
        B = prob.shape[0]

        prob_flat = prob.reshape(B, -1)
        tgt_flat = targets.reshape(B, -1)

        intersection = (prob_flat * tgt_flat).sum(dim=1)
        denominator = prob_flat.sum(dim=1) + tgt_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


class SparsityLoss(nn.Module):
    """
    L1 sparsity penalty on predicted occupancy probability.

    Penalizes the model when its predicted occupancy ratio deviates
    from the target sparsity ratio observed in the training data.
    This prevents the model from collapsing to predicting everything
    occupied or everything free.

        L_sparse = |mean(p) - target_ratio|
    """

    def __init__(self, target_ratio: float = 0.05) -> None:
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prob: (B, ...) predicted occupancy probability

        Returns:
            loss: scalar
        """
        predicted_ratio = prob.mean()
        return (predicted_ratio - self.target_ratio).abs()


class OccupancyLoss(nn.Module):
    """Combined occupancy loss: focal BCE + Lovasz + Dice + Sparsity."""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        dice_weight: float = 0.5,
        sparsity_weight: float = 0.1,
        sparsity_target: float = 0.05,
    ) -> None:
        super().__init__()
        self.focal = FocalBCELoss(focal_alpha, focal_gamma)
        self.lovasz = LovaszLoss()
        self.dice = DiceLoss()
        self.sparsity = SparsityLoss(sparsity_target)

        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.sparsity_weight = sparsity_weight

    def forward(
        self,
        logits: torch.Tensor,
        prob: torch.Tensor,
        targets: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None,
    ) -> dict:
        focal = self.focal(logits, targets, voxel_weight)
        lovasz = self.lovasz(logits, targets)
        dice = self.dice(prob, targets)
        sparsity = self.sparsity(prob)

        total = (
            self.focal_weight * focal
            + self.lovasz_weight * lovasz
            + self.dice_weight * dice
            + self.sparsity_weight * sparsity
        )

        return {
            "focal": focal,
            "lovasz": lovasz,
            "dice": dice,
            "sparsity": sparsity,
            "occupancy_total": total,
        }
