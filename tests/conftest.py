import pytest
import torch
import torch.nn as nn

# Define fixtures that can be used across multiple test modules


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    torch.manual_seed(42)  # For reproducibility
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    return model


@pytest.fixture
def test_data():
    """Generate test data for model training."""
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(5, 10)
    y = torch.tensor([0, 1, 0, 1, 0])
    return x, y


# Configuration for skipping Lightning tests if not available
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "lightning: tests that require PyTorch Lightning")


def pytest_collection_modifyitems(config, items):
    """Skip tests that require Lightning if it's not available."""
    try:
        import lightning
    except ImportError:
        skip_lightning = pytest.mark.skip(reason="PyTorch Lightning not installed")
        for item in items:
            if "lightning" in item.keywords:
                item.add_marker(skip_lightning)
