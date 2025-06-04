import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from zclip import is_fsdp_model


class SimpleModel(nn.Module):
    """A simple model for testing is_fsdp_model function."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class NestedModel(nn.Module):
    """A model with nested modules for testing is_fsdp_model function."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU()
        )
        self.classifier = nn.Linear(5, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class TestIsFSDPModel:
    """Test suite for is_fsdp_model function."""

    def test_non_fsdp_model(self):
        """Test that is_fsdp_model returns False for regular models."""
        model = SimpleModel()
        assert not is_fsdp_model(model)

    @pytest.mark.skipif(not torch.distributed.is_available(), reason="Distributed not available")
    def test_direct_fsdp_model(self, monkeypatch):
        """Test that is_fsdp_model returns True for direct FSDP models."""
        # Mock FSDP to avoid actual distributed setup
        class MockFSDP(nn.Module):
            pass

        # Patch the isinstance check to recognize our mock as FSDP
        original_isinstance = isinstance

        def mock_isinstance(obj, class_or_tuple):
            if class_or_tuple is FSDP and type(obj) is MockFSDP:
                return True
            return original_isinstance(obj, class_or_tuple)

        monkeypatch.setattr(__builtins__, 'isinstance', mock_isinstance)

        # Create a mock FSDP model
        model = MockFSDP()
        
        # Test the function
        assert is_fsdp_model(model)

    @pytest.mark.skipif(not torch.distributed.is_available(), reason="Distributed not available")
    def test_nested_fsdp_model(self, monkeypatch):
        """Test that is_fsdp_model returns True for models with nested FSDP modules."""
        # Create a model with nested modules
        model = NestedModel()
        
        # Mock FSDP to avoid actual distributed setup
        class MockFSDP(nn.Module):
            pass

        # Replace one of the submodules with a mock FSDP module
        model.feature_extractor = MockFSDP()
        
        # Patch the isinstance check
        original_isinstance = isinstance

        def mock_isinstance(obj, class_or_tuple):
            if class_or_tuple is FSDP and type(obj) is MockFSDP:
                return True
            return original_isinstance(obj, class_or_tuple)

        monkeypatch.setattr(__builtins__, 'isinstance', mock_isinstance)
        
        # Test the function
        assert is_fsdp_model(model)
