import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD

from zclip import ZClip


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TestZClipIntegration:
    """Integration tests for ZClip with a real training loop."""
    
    @pytest.fixture
    def setup_training(self):
        """Set up a simple training scenario with synthetic data."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Create synthetic dataset
        n_samples = 100
        n_features = 10
        X = torch.randn(n_samples, n_features)
        # Create a task with clear decision boundary for quick convergence
        y = torch.zeros(n_samples, dtype=torch.long)
        y[X[:, 0] > 0] = 1  # Simple rule: if first feature > 0, class is 1
        
        # Create data loaders
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create model and optimizer
        model = SmallMLP()
        optimizer = SGD(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        
        return model, optimizer, criterion, dataloader

    def test_training_with_zclip(self, setup_training):
        """Test that ZClip works properly in a real training loop."""
        model, optimizer, criterion, dataloader = setup_training
        
        # Create ZClip with small warmup to speed up test
        zclip = ZClip(warmup_steps=2, alpha=0.95)
        
        # Training loop
        n_epochs = 3
        all_losses = []
        all_grad_norms = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for X, y in dataloader:
                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Apply ZClip and get gradient norm
                grad_norm = zclip.step(model)
                
                # Update weights
                optimizer.step()
                
                # Record metrics
                epoch_losses.append(loss.item())
                all_grad_norms.append(grad_norm)
            
            # Record average loss for the epoch
            all_losses.append(sum(epoch_losses) / len(epoch_losses))
        
        # Check that training is working (loss is decreasing)
        assert all_losses[-1] < all_losses[0], "Training did not reduce loss"
        
        # After warmup, ZClip should be initialized
        assert zclip.initialized, "ZClip was not initialized after warmup"
        assert zclip.mean is not None, "ZClip mean was not initialized"
        assert zclip.var is not None, "ZClip variance was not initialized"
        
        # Check that gradient norms are being tracked
        assert len(all_grad_norms) > 0, "Gradient norms were not recorded"
        
        # Ensure gradients are not zero (training is happening)
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Some gradients are None"

    def test_compare_with_without_zclip(self, setup_training):
        """Compare training with and without ZClip to ensure it doesn't break training."""
        model, optimizer, criterion, dataloader = setup_training
        
        # Create a copy of the model for training without ZClip
        model_no_clip = SmallMLP()
        # Initialize with same weights
        model_no_clip.load_state_dict(model.state_dict())
        optimizer_no_clip = SGD(model_no_clip.parameters(), lr=0.05)
        
        # Create ZClip with default settings
        zclip = ZClip(warmup_steps=2)
        
        # Training loop with ZClip
        n_epochs = 3
        for epoch in range(n_epochs):
            for X, y in dataloader:
                # With ZClip
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                zclip.step(model)
                optimizer.step()
                
                # Without ZClip
                optimizer_no_clip.zero_grad()
                outputs_no_clip = model_no_clip(X)
                loss_no_clip = criterion(outputs_no_clip, y)
                loss_no_clip.backward()
                optimizer_no_clip.step()
        
        # Compare final accuracy of both models
        correct_with_clip = 0
        correct_without_clip = 0
        total = 0
        
        with torch.no_grad():
            for X, y in dataloader:
                # With ZClip
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                correct_with_clip += (predicted == y).sum().item()
                
                # Without ZClip
                outputs_no_clip = model_no_clip(X)
                _, predicted_no_clip = torch.max(outputs_no_clip.data, 1)
                correct_without_clip += (predicted_no_clip == y).sum().item()
                
                total += y.size(0)
        
        accuracy_with_clip = correct_with_clip / total
        accuracy_without_clip = correct_without_clip / total
        
        # The accuracy difference should not be dramatic
        # This is a very simple test; in practice, ZClip should improve stability
        # without significantly harming accuracy
        assert abs(accuracy_with_clip - accuracy_without_clip) < 0.2, \
            f"Accuracy difference too large: with={accuracy_with_clip:.4f}, without={accuracy_without_clip:.4f}"

    def test_clip_option_adaptive_scaling(self, setup_training):
        """Test that adaptive scaling clip option works correctly."""
        model, optimizer, criterion, dataloader = setup_training
        
        # Create ZClip with adaptive scaling and small warmup
        zclip = ZClip(
            warmup_steps=2, 
            mode="zscore", 
            clip_option="adaptive_scaling",
            z_thresh=1.5  # Lower threshold to trigger more clipping
        )
        
        # Training loop with monitoring
        clipped_steps = 0
        total_steps = 0
        
        # Run a few batches for testing
        for i, (X, y) in enumerate(dataloader):
            if i >= 5:  # Limit to a few batches for testing speed
                break
                
            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # For testing, spy on the _compute_clip_val method
            original_compute_clip_val = zclip._compute_clip_val
            
            def spy_compute_clip_val(grad_norm):
                nonlocal clipped_steps
                clip_val = original_compute_clip_val(grad_norm)
                if clip_val is not None:
                    clipped_steps += 1
                return clip_val
                
            zclip._compute_clip_val = spy_compute_clip_val
            
            # Apply ZClip
            zclip.step(model)
            
            # Restore original method
            zclip._compute_clip_val = original_compute_clip_val
            
            # Update weights
            optimizer.step()
            total_steps += 1
        
        # After warmup, at least some steps should have been clipped with the low threshold
        if total_steps > 2:  # Skip warmup steps
            clipping_rate = clipping_rate = clipped_steps / (total_steps - 2) if total_steps > 2 else 0
            # With z_thresh=1.5, we expect clipping in about 13% of steps for normal distribution
            # But this is approximate and depends on the actual gradient distribution
            assert clipping_rate < 1.0, "All steps were clipped, which is suspicious"
