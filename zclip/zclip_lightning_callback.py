# zclip_callback.py

from zclip import ZClip

try:
    import lightning as L
except ImportError:
    raise ImportError("PyTorch Lightning is required to use ZClipLightningCallback.")


class ZClipLightningCallback(L.Callback):
    """
    PyTorch Lightning callback for ZClip.
    Applies adaptive gradient clipping after backward pass.
    """
    def __init__(self, **zclip_kwargs):
        super().__init__()
        self.zclip = ZClip(**zclip_kwargs)

    def on_after_backward(self, trainer, pl_module):
        self.zclip.step(pl_module)
