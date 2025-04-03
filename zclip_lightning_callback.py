# zclip_callback.py

import torch
from zclip import ZClip

try:
    import pytorch_lightning as pl
except ImportError:
    raise ImportError("PyTorch Lightning is required to use ZClipCallback.")


class ZClipLightningCallback(pl.Callback):
    """
    PyTorch Lightning callback for ZClip.
    Applies adaptive gradient clipping after backward pass.
    """
    def __init__(self, **zclip_kwargs):
        super().__init__()
        self.zclip = ZClip(**zclip_kwargs)

    def on_after_backward(self, trainer, pl_module):
        self.zclip.step(pl_module)
