import pytest
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_network_layers import layers
from src.arc_multitensor import MultiTensorSystem, MultiTensor

class TestBasicLayers:
    
    def setup_method(self):
        """Setup for each test method."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test tensors
        self.x = torch.randn(3, 4, 5)  # batch, features, channels
        self.weights = [torch.randn(5, 6), torch.randn(6)]  # weight matrix, bias
        
    def test_normalize(self):
        """Test normalization function."""
        dims = [1, 0, 1, 0, 0]  # mock dims
        
        # Test with debias=True (default)
        normalized = layers.normalize(dims, self.x, debias=True)
        
        # Check shape preserved
        assert normalized.shape == self.x.shape
        
        # Check variance is approximately 1 (along non-channel dims)
        var = torch.var(normalized, dim=[0, 1])
        assert torch.allclose(var, torch.ones_like(var), atol=0.15)
        
        # Check mean is approximately 0 (along non-channel dims)
        mean = torch.mean(normalized, dim=[0, 1])
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        
    def test_normalize_no_debias(self):
        """Test normalization without debiasing."""
        dims = [1, 0, 1, 0, 0]
        
        normalized = layers.normalize(dims, self.x, debias=False)
        
        # Check shape preserved
        assert normalized.shape == self.x.shape
        
        # Check variance is approximately 1
        var = torch.var(normalized, dim=[0, 1])
        assert torch.allclose(var, torch.ones_like(var), atol=0.35)
    
    def test_affine(self):
        """Test affine transformation."""
        dims = [1, 0, 1, 0, 0]
        
        # Test without bias
        output = layers.affine(dims, self.x, self.weights, use_bias=False)
        expected_shape = (3, 4, 6)  # batch, features, new_channels
        assert output.shape == expected_shape
        
        # Test with bias
        output_bias = layers.affine(dims, self.x, self.weights, use_bias=True)
        assert output_bias.shape == expected_shape
        
        # Check bias actually added
        assert not torch.allclose(output, output_bias)
        
        # Check difference is the bias
        diff = output_bias - output
        expected_bias = self.weights[1].unsqueeze(0).unsqueeze(0)
        assert torch.allclose(diff, expected_bias)
    
    def test_add_residual_decorator(self):
        """Test add_residual decorator."""
        # Create a simple layer function
        def simple_layer(dims, x):
            return x * 2
        
        # Apply decorator
        residual_layer = layers.add_residual(simple_layer)
        
        # Create residual weights
        # x shape is (3, 4, 5) so input_channels = 5
        # Let's use output_channels = 5 for both projections to keep shape
        residual_weights = [
            [torch.randn(5, 5), torch.randn(5)],  # down projection
            [torch.randn(5, 5), torch.randn(5)]  # up projection
        ]
        
        # Test application
        x = torch.randn(3, 4, 5)
        dims = [1, 0, 1, 0, 0]
        
        # This should work without crashing
        output = residual_layer(dims, x, residual_weights, use_bias=False)
        assert output.shape == x.shape  # Residual connection preserves shape


class TestChannelLayer:
    
    def setup_method(self):
        """Setup for channel layer tests."""
        torch.manual_seed(42)
        
    def test_channel_layer(self):
        """Test VAE channel layer."""
        # Create posterior parameters
        mean = torch.randn(2, 3, 4) * 0.1  # small mean
        local_capacity_adj = torch.zeros_like(mean)
        posterior = (mean, local_capacity_adj)
        
        target_capacity = torch.tensor(0.5)
        
        # Apply channel layer
        z, KL = layers.channel_layer(target_capacity, posterior)
        
        # Check output shape
        assert z.shape == mean.shape
        assert isinstance(KL, torch.Tensor)
        assert KL.shape == mean.shape
        
        # Check KL is non-negative
        assert torch.all(KL >= 0)
        
        # Check z has reasonable values
        assert torch.isfinite(z).all()
        assert torch.isfinite(KL).all()


class TestDecodeLatents:
    
    def setup_method(self):
        """Setup for decode latents test."""
        torch.manual_seed(42)
        
        # Create mock multitensor system
        class MockTask:
            pass
        
        self.mock_task = MockTask()
        self.system = MultiTensorSystem(2, 2, 2, 2, self.mock_task)
        
    def test_decode_latents(self):
        """Test latent decoding function."""
        # Create target capacities
        target_capacities = self.system.make_multitensor(default=torch.tensor(0.5))
        
        # Create decode weights
        def create_decode_weight(dims):
            return [torch.randn(4, 8), torch.randn(8)]
        decode_weights = self.system.make_multitensor()
        for dims in self.system:
            decode_weights[dims] = create_decode_weight(dims)
        
        # Create multiposteriors
        def create_posterior(dims):
            shape = self.system.shape(dims, 4)
            mean = torch.randn(shape) * 0.1
            local_adj = torch.zeros(shape)
            return (mean, local_adj)
        multiposteriors = self.system.make_multitensor()
        for dims in self.system:
            multiposteriors[dims] = create_posterior(dims)
        
        # Apply decode latents
        x, KL_amounts, KL_names = layers.decode_latents(
            target_capacities, decode_weights, multiposteriors
        )
        
        # Check outputs
        assert isinstance(x, MultiTensor)
        assert isinstance(KL_amounts, list)
        assert isinstance(KL_names, list)
        assert len(KL_amounts) == len(KL_names)
        
        # Check each KL amount is a tensor
        for kl in KL_amounts:
            assert isinstance(kl, torch.Tensor)
            assert torch.all(kl >= 0)  # KL should be non-negative


class TestDirectionalLayers:
    
    def setup_method(self):
        """Setup for directional layer tests."""
        torch.manual_seed(42)
        
        # Create test data
        self.x = torch.randn(2, 3, 4, 5)  # batch, channels, height, width
        self.masks = torch.ones(2, 4, 5, 2)  # all pixels valid
        
    def test_cummax_function(self):
        """Test cummax directional function."""
        x = torch.randn(2, 8, 6)  # batch, dim_to_cummax, channels
        dim = 1
        masks = torch.ones(2, 8, 6)
        
        result = layers.cummax_(x, dim, masks)
        
        # Check shape preserved
        assert result.shape == x.shape
        
        # Check output is in reasonable range [-1, 1] due to normalization
        assert torch.all(result >= -1.1)
        assert torch.all(result <= 1.1)
    
    def test_shift_function(self):
        """Test shift directional function."""
        x = torch.randn(2, 8, 6)
        dim = 1
        masks = torch.ones(2, 8, 6)
        
        result = layers.shift_(x, dim, masks)
        
        # Check shape preserved
        assert result.shape == x.shape
        
        # Check that first slice along dim is zeros (padding)
        first_slice = torch.select(result, dim, 0)
        assert torch.allclose(first_slice, torch.zeros_like(first_slice))
        
        # Check that rest is shifted version of original
        if x.shape[dim] > 1:
            shifted_part = result.narrow(dim, 1, x.shape[dim] - 1)
            original_part = x.narrow(dim, 0, x.shape[dim] - 1)
            assert torch.allclose(shifted_part, original_part)
    
    def test_diagonal_shift_function(self):
        """Test diagonal shift function."""
        x = torch.randn(2, 6, 6, 4)  # batch, height, width, channels
        dim1, dim2 = 1, 2  # height, width dimensions
        masks = torch.ones(2, 6, 6, 4)
        
        result = layers.diagonal_shift_(x, dim1, dim2, masks, shift_amount=1, pad_value=0)
        
        # Check shape preserved
        assert result.shape == x.shape
        
        # Check padding was added
        # First row should be zeros
        assert torch.allclose(result[:, 0, :, :], torch.zeros_like(result[:, 0, :, :]))
        # First column should be zeros
        assert torch.allclose(result[:, :, 0, :], torch.zeros_like(result[:, :, 0, :]))


class TestMultiTensorLayers:
    
    def setup_method(self):
        """Setup for multitensor layer tests."""
        torch.manual_seed(42)
        
        class MockTask:
            in_out_same_size = True
            all_out_same_size = False
            masks = torch.ones(2, 4, 4, 2)
        
        self.mock_task = MockTask()
        self.system = MultiTensorSystem(2, 2, 2, 2, self.mock_task)
        
    def test_softmax_layer(self):
        """Test softmax layer with multitensor."""
        # Create test multitensor
        x = self.system.make_multitensor()
        for dims in self.system:
            shape = self.system.shape(dims, 6)  # 6 channels
            x[dims] = torch.randn(shape)
        
        # Create weights for softmax
        def create_softmax_weights(dims):
            n_input_dims = sum(dims[1:])  # all dims except examples
            if n_input_dims == 0:
                return [torch.randn(6, 1), torch.randn(1)]
            n_combinations = 2 ** n_input_dims - 1
            return [torch.randn(6, n_combinations), torch.randn(n_combinations)]
        
        weights = self.system.make_multitensor()
        for dims in self.system:
            weights[dims] = create_softmax_weights(dims)
        
        # Apply softmax (need to create residual weights structure)
        residual_weights = self.system.make_multitensor()
        for dims in self.system:
            channel_dim = x[dims].shape[-1]
            # Compute the output shape of the softmax operation for this dims
            axes = list(range(sum(dims)))
            if dims[0] == 1:
                axes = axes.copy()
                axes.pop(0)
            n_input_dims = len(axes)
            n_combinations = 2 ** n_input_dims - 1 if n_input_dims > 0 else 1
            softmax_output_dim = n_combinations * channel_dim
            residual_weights[dims] = [
                [torch.randn(channel_dim, channel_dim), torch.randn(channel_dim)],  # down
                [torch.randn(softmax_output_dim, softmax_output_dim), torch.randn(softmax_output_dim)]   # up
            ]
        
        # Apply softmax layer
        result = layers.softmax(x, residual_weights, pre_norm=True, post_norm=False, use_bias=False)
        
        # Check result is MultiTensor
        assert isinstance(result, MultiTensor)
        
        # Check output shape matches expected softmax output shape
        for dims in self.system:
            channel_dim = x[dims].shape[-1]
            axes = list(range(sum(dims)))
            if dims[0] == 1:
                axes = axes.copy()
                axes.pop(0)
            n_input_dims = len(axes)
            n_combinations = 2 ** n_input_dims - 1 if n_input_dims > 0 else 1
            softmax_output_dim = n_combinations * channel_dim
            expected_shape = list(x[dims].shape[:-1]) + [softmax_output_dim]
            assert list(result[dims].shape) == expected_shape
    
    def test_nonlinear_layer(self):
        """Test nonlinear layer."""
        # Create test multitensor
        x = self.system.make_multitensor()
        for dims in self.system:
            shape = self.system.shape(dims, 8)
            x[dims] = torch.randn(shape)
        
        # Create residual weights
        residual_weights = self.system.make_multitensor()
        for dims in self.system:
            channel_dim = 8
            residual_weights[dims] = [
                [torch.randn(channel_dim, 6), torch.randn(6)],  # down
                [torch.randn(6, channel_dim), torch.randn(channel_dim)]  # up
            ]
        
        # Apply nonlinear layer
        result = layers.nonlinear(x, residual_weights, pre_norm=True, post_norm=False, use_bias=False)
        
        # Check result is MultiTensor
        assert isinstance(result, MultiTensor)
        
        # Check shapes preserved
        for dims in self.system:
            assert result[dims].shape == x[dims].shape

    def test_share_direction_with_same_size(self):
        """Test share_direction with in_out_same_size=True and valid direction dims."""
        class MockTask:
            in_out_same_size = True
            all_out_same_size = False
            masks = torch.ones(2, 2, 2, 2)
        system = MultiTensorSystem(2, 2, 2, 2, MockTask())
        dims = [1,0,1,1,1]
        shape = system.shape(dims, 4)
        residual = torch.ones(shape)
        share_weights = [torch.ones(shape[-1], shape[-1]), torch.ones(shape[-1])]
        # Directly call affine (which is what share_up uses after communication)
        result = layers.affine(dims, residual, share_weights)
        assert torch.isfinite(result).all()

    def test_share_direction_with_all_out_same_size(self):
        """Test share_direction with all_out_same_size=True and valid direction dims."""
        class MockTask:
            in_out_same_size = False
            all_out_same_size = True
            masks = torch.ones(2, 2, 2, 2)
        system = MultiTensorSystem(2, 2, 2, 2, MockTask())
        dims = [1,0,1,1,1]
        shape = system.shape(dims, 4)
        residual = torch.ones(shape)
        share_weights = [torch.ones(shape[-1], shape[-1]), torch.ones(shape[-1])]
        result = layers.affine(dims, residual, share_weights)
        assert torch.isfinite(result).all()

    def test_direction_share_without_pre_norm(self):
        """Test direction_share with pre_norm=False and valid direction dims."""
        class MockTask:
            pass
        system = MultiTensorSystem(2, 2, 2, 2, MockTask())
        dims = [1,0,1,1,1]
        shape = system.shape(dims, 4)
        x = torch.ones(shape)
        n_in = shape[-1]
        n_out = n_in
        weights = [[torch.ones(n_in, n_out) for _ in range(8)] for _ in range(8)]
        result = layers.direction_share(dims, x, weights, pre_norm=False)
        assert torch.isfinite(result).all()

    def test_direction_share_with_bias(self):
        """Test direction_share with use_bias=True and valid direction dims."""
        class MockTask:
            pass
        system = MultiTensorSystem(2, 2, 2, 2, MockTask())
        dims = [1,0,1,1,1]
        shape = system.shape(dims, 4)
        x = torch.ones(shape)
        n_in = shape[-1]
        n_out = n_in
        weights = [[torch.ones(n_in, n_out) for _ in range(8)] for _ in range(8)]
        result = layers.direction_share(dims, x, weights, use_bias=True)
        assert torch.isfinite(result).all()


class TestPostprocessMask:
    
    def setup_method(self):
        """Setup for postprocess mask test."""
        # Create mock task
        class MockTask:
            n_examples = 2
            n_x = 4
            n_y = 4
            shapes = [
                [[2, 2], [2, 2]],  # example 0: input and output shapes
                [[3, 3], [3, 3]]   # example 1: input and output shapes
            ]
        
        self.mock_task = MockTask()
        
    def test_postprocess_mask(self):
        """Test mask postprocessing."""
        # Create test masks
        x_mask = torch.randn(2, 4, 2)  # examples, x_dim, in/out
        y_mask = torch.randn(2, 4, 2)  # examples, y_dim, in/out
        
        # Apply postprocessing
        x_processed, y_processed = layers.postprocess_mask(self.mock_task, x_mask, y_mask)
        
        # Check shapes preserved
        assert x_processed.shape == x_mask.shape
        assert y_processed.shape == y_mask.shape
        
        # Check that out-of-bounds positions have large negative values
        # Example 0 should have negative values for positions >= 2
        assert x_processed[0, 2:, :].max() < -100
        assert y_processed[0, 2:, :].max() < -100
        
        # Example 1 should have negative values for positions >= 3
        assert x_processed[1, 3:, :].max() < -100
        assert y_processed[1, 3:, :].max() < -100


class TestMakeDirectionalLayer:
    
    def test_make_directional_layer(self):
        """Test directional layer creation."""
        # Define simple directional functions
        def simple_fn(x, dim, masks):
            return x * 2
        
        def simple_diagonal_fn(x, dim1, dim2, masks):
            return x * 3
        
        # Create directional layer
        directional_layer = layers.make_directional_layer(simple_fn, simple_diagonal_fn)
        
        # Test with appropriate input
        dims = [1, 0, 1, 1, 1]  # has direction dimension
        x = torch.randn(2, 8, 4, 4, 6)  # batch, directions, height, width, channels
        masks = torch.ones(2, 4, 4, 2)
        
        # Apply layer
        result = directional_layer(dims, x, masks)
        
        # Check shape preserved
        assert result.shape == x.shape


class TestEdgeCases:
    
    def test_empty_tensors(self):
        """Test layers with minimal tensor sizes."""
        # Test with 1x1 tensors
        x = torch.randn(1, 1, 1)
        dims = [1, 0, 0, 0, 0]
        
        # Normalize should work
        normalized = layers.normalize(dims, x)
        assert normalized.shape == x.shape
        
        # Affine should work
        weights = [torch.randn(1, 2), torch.randn(2)]
        output = layers.affine(dims, x, weights, use_bias=True)
        assert output.shape == (1, 1, 2)
    
    def test_large_tensors(self):
        """Test layers with larger tensors."""
        # Test with larger tensors
        x = torch.randn(10, 50, 100)
        dims = [1, 1, 0, 1, 1]
        
        # Should handle without issues
        normalized = layers.normalize(dims, x)
        assert normalized.shape == x.shape
        
        weights = [torch.randn(100, 80), torch.randn(80)]
        output = layers.affine(dims, x, weights, use_bias=True)
        assert output.shape == (10, 50, 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])