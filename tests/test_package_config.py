import importlib.util
import pytest


def test_base_package():
    """Test that the base package imports correctly."""
    import zclip
    assert hasattr(zclip, 'ZClip')
    assert hasattr(zclip, 'is_fsdp_model')


def test_lightning_import():
    """Test that the lightning callback imports correctly if lightning is available."""
    has_lightning = importlib.util.find_spec("lightning") is not None
    
    import zclip
    
    if has_lightning:
        # If lightning is available, ZClipLightningCallback should be in the namespace
        assert hasattr(zclip, 'ZClipLightningCallback')
        # Import the callback directly
        from zclip import ZClipLightningCallback
        assert ZClipLightningCallback is not None
    else:
        # If lightning is not available, ZClipLightningCallback should not be in the namespace
        assert not hasattr(zclip, 'ZClipLightningCallback')
        # Importing the callback directly should fail
        with pytest.raises(ImportError):
            from zclip.zclip_lightning_callback import ZClipLightningCallback


if __name__ == "__main__":
    test_base_package()
    test_lightning_import()
