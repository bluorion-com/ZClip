import pytest
import torch
import torch.nn as nn

try:
    import lightning as L
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks import Callback
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pytest.skip("Lightning not available", allow_module_level=True)

from zclip import ZClipLightningCallback


@pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="Lightning not available")
class SimpleModel(LightningModule):
    """A simple Lightning model for testing ZClip callback."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="Lightning not available")
class MockCallback(Callback):
    """Mock callback to capture when ZClip callback is called."""

    def __init__(self):
        super().__init__()
        self.optimizer_step_called = False
        self.zclip_called_after_backward = False
        self.zclip_called_before_optimizer = False
        self.zclip_step_model = None

    def on_after_backward(self, trainer, pl_module):
        # Check if ZClip has been called at this point
        for callback in trainer.callbacks:
            if isinstance(callback, ZClipLightningCallback):
                if hasattr(callback, "_called"):
                    self.zclip_called_after_backward = True
                self.zclip_step_model = pl_module

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Check if ZClip has been called at this point
        for callback in trainer.callbacks:
            if isinstance(callback, ZClipLightningCallback):
                if hasattr(callback, "_called"):
                    self.zclip_called_before_optimizer = True

    def on_optimizer_step(self, trainer, pl_module, optimizer):
        self.optimizer_step_called = True


@pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="Lightning not available")
class TestZClipLightningCallback:
    """Test suite for ZClipLightningCallback."""

    @pytest.fixture
    def patched_zclip_callback(self, monkeypatch):
        """Patch the ZClipLightningCallback to track calls."""
        original_step = ZClipLightningCallback.on_before_optimizer_step

        def patched_on_before_optimizer_step(self, trainer, pl_module, optimizer):
            self._called = True
            return original_step(self, trainer, pl_module, optimizer)

        monkeypatch.setattr(ZClipLightningCallback, 'on_before_optimizer_step', patched_on_before_optimizer_step)
        return ZClipLightningCallback()

    @pytest.fixture
    def model_and_batch(self):
        """Create a model and a batch for testing."""
        torch.manual_seed(42)
        model = SimpleModel()
        x = torch.randn(5, 10)
        y = torch.tensor([0, 1, 0, 1, 0])
        return model, (x, y)

    def test_callback_initialization(self):
        """Test that ZClipLightningCallback initializes correctly."""
        callback = ZClipLightningCallback(alpha=0.9, z_thresh=3.0)
        
        assert callback.zclip.alpha == 0.9
        assert callback.zclip.z_thresh == 3.0
        assert callback.zclip.mode == "zscore"  # Default value

    @pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="Lightning not available")
    def test_callback_called_before_optimizer(self, patched_zclip_callback, model_and_batch):
        """Test that ZClipLightningCallback is called at the right time in the training loop."""
        model, batch = model_and_batch
        mock_callback = MockCallback()
        
        trainer = Trainer(
            max_steps=1,  # Just run one step for testing
            callbacks=[patched_zclip_callback, mock_callback],
            enable_checkpointing=False,  # Disable checkpointing for test speed
            logger=False,  # Disable logging for test speed
            accelerator="cpu"
        )
        
        # Run one training step
        trainer.fit(model, torch.utils.data.DataLoader([batch], batch_size=1))
        
        # Check that the callbacks were called in the right order
        assert mock_callback.optimizer_step_called
        assert not mock_callback.zclip_called_after_backward  # Should not be called after backward
        assert mock_callback.zclip_called_before_optimizer  # Should be called before optimizer step

    def test_zclip_step_called_with_model(self, monkeypatch):
        """Test that ZClip's step method is called with the model."""
        called_with = None
        
        # Create a callback with a mocked ZClip
        callback = ZClipLightningCallback()
        
        def mock_step(self, model):
            nonlocal called_with
            called_with = model
            return 1.0  # Return a dummy gradient norm
            
        monkeypatch.setattr(callback.zclip, 'step', mock_step.__get__(callback.zclip))
        
        # Create a dummy model and optimizer
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Call the callback
        callback.on_before_optimizer_step(None, model, optimizer)
        
        # Check that ZClip.step was called with the model
        assert called_with is model


if __name__ == "__main__":
    pytest.main()
