# zclip_callback.py

from zclip import ZClip

try:
    import lightning as L
except ImportError:
    raise ImportError(
        "PyTorch Lightning is required to use ZClipLightningCallback. "
        "Please install it with: pip install 'zclip[lightning]'"
    )


class ZClipLightningCallback(L.Callback):
    """
    PyTorch Lightning callback for ZClip.
    Applies adaptive gradient clipping before optimizer step.
    """
    def __init__(self, **zclip_kwargs):
        super().__init__()
        self.zclip = ZClip(**zclip_kwargs)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.zclip.step(pl_module)
