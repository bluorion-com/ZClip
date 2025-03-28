# zclip.py

import torch
import scipy.stats

class ZClip:
    def __init__(self, alpha=0.97, z_thresh=2.5, max_grad_norm=None, eps=1e-6,
                 warmup_steps=25, mode="zscore", percentile=0.99):
        """
        ZClip: An adaptive gradient clipping mechanism using EMA and anomaly detection.

        Args:
            alpha (float): EMA smoothing factor for mean and variance.
            z_thresh (float): Z-score threshold to trigger adaptive clipping (used only in 'zscore' mode).
            max_grad_norm (float or None): Optional max gradient norm to apply on top of adaptive clipping.
                                           If None (default), max norm clipping is not enabled.
            eps (float): Small constant to avoid division by zero.
            warmup_steps (int): Number of initial steps to collect gradient norms before EMA is initialized.
            mode (str): One of {"zscore", "percentile"}. Determines the clipping logic.
            percentile (float): Percentile value in range (0, 1) used in 'percentile' mode.
        """
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.mode = mode.lower()
        self.percentile = percentile

        assert self.mode in ["zscore", "percentile"], "Mode must be 'zscore' or 'percentile'."
        if self.mode == "percentile":
            assert 0 < self.percentile < 1, "percentile must be between 0 and 1."
            self.z_thresh = scipy.stats.norm.ppf(self.percentile)

        self.buffer = []
        self.initialized = False
        self.mean = None
        self.var = None

    def _initialize_ema(self):
        self.mean = sum(self.buffer) / len(self.buffer)
        self.var = sum((x - self.mean) ** 2 for x in self.buffer) / len(self.buffer)
        self.initialized = True
        self.buffer = []

    def _update_ema(self, grad_norm):
        # Update EMA for mean and variance using the new effective gradient norm.
        self.mean = self.alpha * self.mean + (1 - self.alpha) * grad_norm
        self.var = self.alpha * self.var + (1 - self.alpha) * (grad_norm - self.mean) ** 2

    def _compute_zscore(self, grad_norm):
        std = self.var ** 0.5
        z = (grad_norm - self.mean) / (std + self.eps)
        return z, std

    def _compute_grad_norm(self, model):
        # Compute the total L2 norm of all gradients in the model.
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if not grad_norms:
            return 0.0
        grad_norms_tensor = torch.stack(grad_norms)
        total_norm = torch.sqrt(torch.sum(grad_norms_tensor ** 2))
        return total_norm.item()

    def _compute_clip_val(self, grad_norm):
        std = self.var ** 0.5
        if self.mode == "zscore":
            z, std = self._compute_zscore(grad_norm)
            if z > self.z_thresh:
                eta = z / self.z_thresh
                threshold = self.mean + (self.z_thresh * std) / eta
                return threshold
        elif self.mode == "percentile":
            threshold = self.mean + self.z_thresh * std
            if grad_norm > threshold:
                return threshold
        return None  # No adaptive clipping needed

    def _apply_clipping(self, model, clip_val, total_norm):
        """
        Applies clipping to the gradients by merging the adaptive clip value with the optional max_grad_norm.
        """
        # Use the adaptive clip if computed; otherwise fall back to the total norm.
        adaptive_clip = clip_val if clip_val is not None else total_norm
        if self.max_grad_norm is not None:
            effective_clip = min(adaptive_clip, self.max_grad_norm)
        else:
            effective_clip = adaptive_clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), effective_clip)
        return effective_clip

    def step(self, model):
        """
        Call this after loss.backward() but before optimizer.step().

        Args:
            model (torch.nn.Module): The model with computed gradients.
        
        Returns:
            float: The total gradient norm (before any clipping) for monitoring.
        """
        total_norm = self._compute_grad_norm(model)

        # During warmup, collect gradient norms without applying adaptive clipping.
        if not self.initialized:
            self.buffer.append(total_norm)
            if len(self.buffer) >= self.warmup_steps:
                self._initialize_ema()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            return total_norm

        # Compute the adaptive clip value.
        clip_val = self._compute_clip_val(total_norm)
        # Apply clipping via the helper method.
        self._apply_clipping(model, clip_val, total_norm)
        # Update EMA using the adaptive clip value (or total_norm if no clipping was triggered).
        self._update_ema(clip_val if clip_val is not None else total_norm)
        return total_norm
