import torch
import torch.nn.functional as F
import numpy as np


class UncertaintyAwareDualTraining:
    """
    Uncertainty-aware Dual-branch Training (UDT) Strategy

    Key features:
        - Uncertainty estimation from dual-branch disagreement and entropy
        - Frequency-domain uncertainty enhancement using Haar features
        - Adaptive sample weighting based on uncertainty
        - Dual-branch consistency regularization
        - Progressive thresholding for reliable samples

    Args:
        num_classes: Number of segmentation classes
        consistency_weight: Weight for branch disagreement in uncertainty
        entropy_weight: Weight for prediction entropy in uncertainty
        haar_weight: Weight for frequency-domain uncertainty
        uncertainty_loss_weight: Weight for uncertainty-weighted CE loss
        consistency_loss_weight: Weight for dual-branch consistency loss
        base_threshold: Confidence threshold for reliable samples
        warmup_epochs: Number of warmup epochs before applying threshold
        weight_type: Type of uncertainty weighting ['exp', 'square', 'linear']
        device: Computation device
        verbose: Whether to print debug information
    """

    def __init__(self,
                 num_classes: int,
                 consistency_weight: float = 0.4,
                 entropy_weight: float = 0.4,
                 haar_weight: float = 0.2,
                 uncertainty_loss_weight: float = 0.2,
                 consistency_loss_weight: float = 0.1,
                 base_threshold: float = 0.7,
                 warmup_epochs: int = 5,
                 weight_type: str = "exp",
                 device: str = 'cuda',
                 verbose: bool = False):

        self.num_classes = num_classes
        self.device = device
        self.verbose = verbose

        # Uncertainty estimation weights
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight
        self.haar_weight = haar_weight

        # Loss weights
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.consistency_loss_weight = consistency_loss_weight

        # Progressive thresholding
        self.base_threshold = base_threshold
        self.warmup_epochs = warmup_epochs

        # Weighting strategy
        self.weight_type = weight_type

        # Training state
        self.current_epoch = 0

        # Statistics tracking
        self.uncertainty_history = []
        self.consistency_history = []

        if verbose:
            print(f"✅ UDT initialized (weight_type={weight_type})")

    # ==================== Uncertainty Estimation ====================

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range"""
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    def _compute_haar_uncertainty(self, haar_features: list, target_size) -> torch.Tensor:
        """
        Compute frequency-domain uncertainty from Haar features.
        High-frequency components often indicate challenging regions.
        """
        if not haar_features:
            return None

        # Use the deepest Haar feature for frequency analysis
        feat = haar_features[-1]

        # Align spatial dimensions
        if feat.shape[2:] != target_size:
            feat = F.interpolate(feat, size=target_size,
                                 mode='bilinear', align_corners=False)

        # Compute gradients (high-frequency components)
        grad_x = torch.abs(feat[:, :, :, 1:] - feat[:, :, :, :-1]).mean(dim=1)
        grad_y = torch.abs(feat[:, :, 1:, :] - feat[:, :, :-1, :]).mean(dim=1)

        # Pad to maintain spatial dimensions
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        return (grad_x + grad_y) / 2

    def estimate_uncertainty(self,
                             local_probs: torch.Tensor,
                             global_probs: torch.Tensor,
                             haar_features: list = None) -> tuple:
        """
        Estimate pixel-wise uncertainty from dual-branch predictions.

        Args:
            local_probs: Probability maps from local branch (ConvNeXt)
            global_probs: Probability maps from global branch (Swin)
            haar_features: List of Haar-enhanced features for frequency analysis

        Returns:
            uncertainty: Uncertainty map [B, H, W]
            confidence: Confidence map [B, H, W]
        """
        # 1. Branch disagreement
        disagreement = torch.abs(local_probs - global_probs).mean(dim=1)

        # 2. Prediction entropy (using fused predictions)
        fused_probs = (local_probs + global_probs) / 2
        entropy = -torch.sum(fused_probs * torch.log(fused_probs + 1e-10), dim=1)
        max_entropy = torch.log(torch.tensor(self.num_classes, device=self.device))
        normalized_entropy = entropy / max_entropy

        # Combine disagreement and entropy
        uncertainty = (self.consistency_weight * disagreement +
                       self.entropy_weight * normalized_entropy)

        # 3. Frequency-domain uncertainty (Haar features)
        if haar_features is not None:
            haar_uncertainty = self._compute_haar_uncertainty(
                haar_features, uncertainty.shape[1:]
            )
            if haar_uncertainty is not None:
                uncertainty = uncertainty + self.haar_weight * haar_uncertainty

        # Normalize to [0, 1]
        uncertainty = self._normalize(uncertainty).clamp(0, 1)
        confidence = 1 - uncertainty

        return uncertainty, confidence

    # ==================== Uncertainty Weighting ====================

    def _get_weight(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Convert uncertainty to sample weight.
        Higher confidence (lower uncertainty) yields higher weight.

        Supported weighting strategies:
            - 'exp': Exponential decay (smooth, recommended)
            - 'square': Quadratic decay (aggressive)
            - 'linear': Linear decay (baseline)
        """
        if self.weight_type == "exp":
            return torch.exp(-uncertainty)
        elif self.weight_type == "square":
            return (1 - uncertainty) ** 2
        else:  # linear
            return 1 - uncertainty

    # ==================== Loss Computation ====================

    def compute_consistency_loss(self,
                                 local_logits: torch.Tensor,
                                 global_logits: torch.Tensor,
                                 uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute dual-branch consistency loss with uncertainty weighting.

        Encourages agreement between local and global branches,
        with lower penalty on uncertain regions.
        """
        # Align spatial dimensions
        if local_logits.shape[2:] != global_logits.shape[2:]:
            global_logits = F.interpolate(
                global_logits,
                size=local_logits.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Convert to probability distributions
        local_probs = F.softmax(local_logits, dim=1)
        global_probs = F.softmax(global_logits, dim=1)

        # Symmetric KL divergence
        kl_local_to_global = F.kl_div(local_probs.log(), global_probs,
                                      reduction='none').sum(1)
        kl_global_to_local = F.kl_div(global_probs.log(), local_probs,
                                      reduction='none').sum(1)
        divergence = (kl_local_to_global + kl_global_to_local) / 2

        # Apply uncertainty weighting
        weight = self._get_weight(uncertainty)
        loss = (divergence * weight).mean()

        return self.consistency_loss_weight * loss

    def compute_uncertainty_loss(self,
                                 logits: torch.Tensor,
                                 target: torch.Tensor,
                                 uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss with uncertainty-aware weighting.

        Down-weights unreliable samples to reduce noise in training.
        """
        # Pixel-wise cross-entropy
        ce_loss = F.cross_entropy(logits, target, reduction='none')

        # Uncertainty weighting
        weight = self._get_weight(uncertainty)

        # Progressive thresholding (after warmup)
        if self.current_epoch >= self.warmup_epochs:
            confidence = 1 - uncertainty
            reliable_mask = (confidence > self.base_threshold).float()
            weight = weight * reliable_mask

        # Weighted average
        weighted_loss = (ce_loss * weight).sum() / (weight.sum() + 1e-6)

        return self.uncertainty_loss_weight * weighted_loss

    # ==================== Main Forward ====================

    def forward(self,
                model,
                images: torch.Tensor,
                targets: torch.Tensor) -> tuple:
        """
        Perform one forward pass with UDT supervision.

        Args:
            model: DUHA-Net model (must support forward_with_features)
            images: Input images [B, C, H, W]
            targets: Ground truth labels [B, H, W]

        Returns:
            total_loss: Combined loss
            uncertainty: Uncertainty map (for monitoring)
            confidence: Confidence map (for monitoring)
            predictions: Segmentation output
        """
        # Forward pass with features
        predictions, features = model.forward_with_features(images)

        # Extract main output
        if isinstance(predictions, (tuple, list)):
            main_output = predictions[0]
        else:
            main_output = predictions

        # Base segmentation loss
        total_loss = F.cross_entropy(main_output, targets)

        # If not in dual-branch mode, return base loss
        if features is None or not hasattr(features, 'local_logits'):
            return total_loss, None, None, main_output

        local_logits = features.local_logits
        global_logits = features.global_logits

        # Convert to probabilities
        local_probs = F.softmax(local_logits, dim=1)
        global_probs = F.softmax(global_logits, dim=1)

        # Estimate uncertainty
        uncertainty, confidence = self.estimate_uncertainty(
            local_probs, global_probs, getattr(features, 'haar_features', None)
        )

        # Uncertainty-weighted loss
        uncertainty_weighted_loss = self.compute_uncertainty_loss(
            main_output, targets, uncertainty
        )
        total_loss = total_loss + uncertainty_weighted_loss

        # Consistency loss
        consistency_loss = self.compute_consistency_loss(
            local_logits, global_logits, uncertainty
        )
        total_loss = total_loss + consistency_loss

        # Track statistics
        self.uncertainty_history.append(uncertainty.mean().item())
        self.consistency_history.append(consistency_loss.item())

        return total_loss, uncertainty, confidence, main_output

    # ==================== Training Control ====================

    def set_epoch(self, epoch: int):
        """Update current epoch for progressive thresholding"""
        self.current_epoch = epoch

    def get_stats(self) -> dict:
        """Get training statistics"""
        return {
            "epoch": self.current_epoch,
            "uncertainty": np.mean(self.uncertainty_history[-100:])
            if self.uncertainty_history else 0,
            "consistency_loss": np.mean(self.consistency_history[-100:])
            if self.consistency_history else 0
        }

    def reset_stats(self):
        """Reset statistics history"""
        self.uncertainty_history = []
        self.consistency_history = []


# ==================== Training Helper ====================

class UDTTrainer:
    """
    Training wrapper for DUHA-Net with UDT strategy.

    Example:
        >>> model = DUHANet(num_classes=6)
        >>> udt = UncertaintyAwareDualTraining(num_classes=6)
        >>> trainer = UDTTrainer(model, udt)
        >>>
        >>> for epoch in range(num_epochs):
        ...     trainer.set_epoch(epoch)
        ...     for images, targets in dataloader:
        ...         loss = trainer.train_step(images, targets)
    """

    def __init__(self, model, udt_strategy, optimizer, device='cuda'):
        self.model = model.to(device)
        self.udt = udt_strategy
        self.optimizer = optimizer
        self.device = device

    def set_epoch(self, epoch: int):
        """Set current epoch for progressive thresholding"""
        self.udt.set_epoch(epoch)

    def train_step(self, images, targets):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        images = images.to(self.device)
        targets = targets.to(self.device)

        loss, uncertainty, confidence, predictions = self.udt.forward(
            self.model, images, targets
        )

        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'uncertainty': uncertainty.mean().item() if uncertainty is not None else 0,
            'stats': self.udt.get_stats()
        }

    @torch.no_grad()
    def eval_step(self, images, targets=None):
        """Single evaluation step"""
        self.model.eval()
        images = images.to(self.device)

        predictions = self.model(images)
        if isinstance(predictions, dict):
            predictions = predictions['segmentation']

        return predictions


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Create model
    from duha_net import DUHANet  # Assume model is imported

    model = DUHANet(num_classes=6, img_size=256)

    # Create UDT strategy
    udt = UncertaintyAwareDualTraining(
        num_classes=6,
        consistency_weight=0.4,
        entropy_weight=0.4,
        haar_weight=0.2,
        uncertainty_loss_weight=0.2,
        consistency_loss_weight=0.1,
        weight_type="exp",
        verbose=True
    )

    # Simulate training
    x = torch.randn(2, 3, 256, 256)
    target = torch.randint(0, 6, (2, 256, 256))

    loss, uncertainty, confidence, pred = udt.forward(model, x, target)

    print(f"Loss: {loss.item():.4f}")
    print(f"Mean uncertainty: {uncertainty.mean().item():.4f}")
    print(f"Stats: {udt.get_stats()}")