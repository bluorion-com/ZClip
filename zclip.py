import torch

class ZClip:
    def __init__(self, alpha=0.97, z_thresh=2.5, max_grad_norm=None, eps=1e-6,
                 warmup_steps=25, mode="zscore", clip_option="adaptive_scaling"):
        """
        ZClip: An adaptive gradient clipping mechanism using EMA and anomaly detection.

        Args:
            alpha (float): EMA smoothing factor for mean and variance.
            z_thresh (float): Threshold value.
                              In percentile mode, the clipping threshold is computed as:
                                  EMA mean + (z_thresh × std)
                              In zscore mode, z_thresh is used to determine whether to clip to the baseline
                              or to compute an adaptive threshold.
            max_grad_norm (float or None): Optional maximum gradient norm.
                                           If None, max norm clipping is not applied.
            eps (float): Small constant to avoid division by zero.
            warmup_steps (int): Number of steps to collect gradient norms before EMA initialization.
            mode (str): Clipping mode. Options:
                        - "percentile": Always clip to a fixed threshold defined as EMA mean plus (z_thresh × std).
                        - "zscore":     Use z-score based clipping.
            clip_option (str): Only used when mode is "zscore". Options:
                        - "adaptive_scaling": If the gradient norm is a strong outlier (z-score > z_thresh),
                                               compute an adaptive threshold as:
                                                   EMA mean + (z_thresh × std) / (z/z_thresh)
                        - "mean": Simply clip to the EMA mean when the z-score exceeds z_thresh.
        """
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.mode = mode.lower()

        if self.mode == "zscore":
            assert clip_option in ["mean", "adaptive_scaling"], (
                "For zscore mode, clip_option must be either 'mean' or 'adaptive_scaling'."
            )
            self.clip_option = clip_option.lower()
        elif self.mode == "percentile":
            self.clip_option = None  # clip_option is ignored in percentile mode.
        else:
            raise ValueError("mode must be either 'zscore' or 'percentile'.")

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

        # Fixed behavior: In percentile mode, always clip to a threshold computed as:
        #   EMA mean + (z_thresh × std)
        if self.mode == "percentile":
            threshold = self.mean + self.z_thresh * std
            if grad_norm > threshold:
                return threshold
        elif self.mode == "zscore":
            # Compute the z-score for the current gradient norm.
            z, std = self._compute_zscore(grad_norm)
            if z > self.z_thresh:
                if self.clip_option == "adaptive_scaling":
                    # Adaptive Scaling: Use an adaptive threshold based on the z-score.
                    eta = z / self.z_thresh
                    threshold = self.mean + (self.z_thresh * std) / eta
                elif self.clip_option == "mean":
                    # Baseline Mean: Simply clip to the EMA mean.
                    threshold = self.mean
                return threshold
        return None  # No clipping needed.

    def _apply_clipping(self, model, clip_val, total_norm):
        """
        Applies clipping to the gradients by merging the computed clip value with the optional max_grad_norm.
        """
        # Use the computed clip_val if available; otherwise, use the total norm.
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
            float: The total gradient norm (before clipping) for monitoring.
        """
        total_norm = self._compute_grad_norm(model)

        # During warmup, collect gradient norms without applying clipping.
        if not self.initialized:
            self.buffer.append(total_norm)
            if len(self.buffer) >= self.warmup_steps:
                self._initialize_ema()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            return total_norm

        # Compute the clip value based on the selected mode and clip_option.
        clip_val = self._compute_clip_val(total_norm)
        self._apply_clipping(model, clip_val, total_norm)
        # Update EMA with the effective norm (either the computed clip or the original norm).
        self._update_ema(clip_val if clip_val is not None else total_norm)
        return total_norm
