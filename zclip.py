# zclip.py

import torch
import scipy.stats

class ZClip:
    def __init__(self, alpha=0.97, z_thresh=2.5, clip_factor=1.0, eps=1e-6,
                 warmup_steps=25, mode="zscore", percentile=0.99):
        """
        ZClip: An adaptive gradient clipping mechanism using EMA and anomaly detection.

        Args:
            alpha (float): EMA smoothing factor for mean and variance.
            z_thresh (float): Z-score threshold to trigger clipping (used only in 'zscore' mode).
            clip_factor (float): Scaling factor for determining the clipping value (zscore only).
            eps (float): Small constant to avoid division by zero.
            warmup_steps (int): Number of initial steps to collect gradient norms before EMA starts.
            mode (str): One of {"zscore", "percentile"}. Chooses clipping logic.
            percentile (float): Percentile value in range (0, 1) used for clipping in 'percentile' mode.
        """
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.clip_factor = clip_factor
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
        self.mean = self.alpha * self.mean + (1 - self.alpha) * grad_norm
        self.var = self.alpha * self.var + (1 - self.alpha) * (grad_norm - self.mean) ** 2

    def _compute_zscore(self, grad_norm):
        std = self.var ** 0.5
        z = (grad_norm - self.mean) / (std + self.eps)
        return z, std

    def compute_grad_norm(model):
        grad_norms = [
            p.grad.norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]
        if not grad_norms:
            return 0.0

        # Stack individual param norms and compute total L2 norm
        grad_norms_tensor = torch.stack(grad_norms)
        total_norm = torch.sqrt(torch.sum(grad_norms_tensor ** 2))
        return total_norm.item()

    def _compute_clip_val(self, grad_norm):
        std = self.var ** 0.5
        if self.mode == "zscore":
            z, std = self._compute_zscore(grad_norm)
            if z > self.z_thresh:
                return self.mean + self.clip_factor * std * (self.z_thresh / z)
        elif self.mode == "percentile":
            threshold = self.mean + self.z_thresh * std
            if grad_norm > threshold:
                return threshold
        return None  # No clipping needed

    def step(self, model):
        """
        Call this after loss.backward() but before optimizer.step().

        Args:
            model (torch.nn.Module): Model with gradients computed.
        """
        total_norm = self.compute_grad_norm(model)

        if not self.initialized:
            self.buffer.append(total_norm)
            if len(self.buffer) >= self.warmup_steps:
                self._initialize_ema()
            return total_norm

        gt_update = total_norm
        clip_val = self._compute_clip_val(total_norm)
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            gt_update = clip_val

        self._update_ema(gt_update)
        return total_norm