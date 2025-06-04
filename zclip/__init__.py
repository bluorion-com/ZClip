from zclip.zclip import ZClip, is_fsdp_model

__all__ = ["ZClip", "is_fsdp_model"]

# Conditionally import Lightning components if available
try:
    from importlib.util import find_spec
    has_lightning = find_spec("lightning") is not None
    
    if has_lightning:
        from zclip.zclip_lightning_callback import ZClipLightningCallback
        __all__ += ["ZClipLightningCallback"]
except ImportError:
    # Lightning is not installed, callback won't be available
    pass
