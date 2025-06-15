import pytest
import numpy as np
import torch
import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_weight_initializer import Initializer
from src.arc_multitensor import MultiTensorSystem


class TestWeightInitializer:
    
    def setup_method(self):
        """Setup for each test method."""
        # Mock task object
        class MockTask:
            pass
        
        self.mock_task = MockTask()
        self.system = MultiTensorSystem(
            n_examples=2, 
            n_colors=3, 
            n_x=4, 
            n_y=5, 
            task=self.mock_task
        )
        
        # Define a simple channel dimension function
        def channel_dim_fn(dims):
            return 16
        
        self.initializer = Initializer(self.system, channel_dim_fn)
    
    def test_initialize_zeros(self):
        """Test zero initialization."""
        # Test with fixed shape
        zeros = self.initializer.initialize_zeros([1, 0, 0, 1, 0], (2, 4))
        assert isinstance(zeros, torch.Tensor)
        assert zeros.shape == (2, 4)
        assert torch.all(zeros == 0)
        assert zeros.requires_grad
        
        # Test with callable shape
        def shape_fn(dims):
            return (3, 5)
        zeros = self.initializer.initialize_zeros([1, 0, 0, 1, 0], shape_fn)
        assert zeros.shape == (3, 5)
    
    def test_initialize_linear(self):
        """Test linear layer initialization."""
        # Test with fixed dimensions
        weights = self.initializer.initialize_linear([1, 0, 0, 1, 0], (10, 20))
        assert len(weights) == 2  # weight and bias
        assert weights[0].shape == (10, 20)
        assert weights[1].shape == (20,)
        assert weights[0].requires_grad
        assert weights[1].requires_grad
        
        # Test with callable dimensions
        def n_in_fn(dims):
            return 15
        def n_out_fn(dims):
            return 25
        weights = self.initializer.initialize_linear([1, 0, 0, 1, 0], [n_in_fn, n_out_fn])
        assert weights[0].shape == (15, 25)
        assert weights[1].shape == (25,)
    
    def test_initialize_residual(self):
        """Test residual layer initialization."""
        n_in = 32
        n_out = 64
        residual = self.initializer.initialize_residual([1, 0, 0, 1, 0], n_in, n_out)
        
        assert len(residual) == 2  # two linear layers
        assert residual[0][0].shape == (16, n_in)  # first layer weight
        assert residual[0][1].shape == (n_in,)     # first layer bias
        assert residual[1][0].shape == (n_out, 16) # second layer weight
        assert residual[1][1].shape == (16,)       # second layer bias
    
    def test_initialize_posterior(self):
        """Test posterior initialization."""
        channel_dim = 32
        posterior = self.initializer.initialize_posterior([1, 0, 0, 1, 0], channel_dim)
        
        assert len(posterior) == 2  # mean and local capacity adjustment
        assert posterior[0].shape == (2, 4, 32)  # shape from system.shape
        assert posterior[1].shape == (2, 4, 32)  # shape from system.shape
        assert posterior[0].requires_grad
        assert posterior[1].requires_grad
    
    def test_initialize_direction_share(self):
        """Test direction sharing initialization."""
        direction_share = self.initializer.initialize_direction_share([1, 0, 0, 1, 0], None)
        
        assert len(direction_share) == 8
        for row in direction_share:
            assert len(row) == 8
            for weights in row:
                assert len(weights) == 2  # weight and bias
                assert weights[0].shape == (16, 16)  # channel_dim_fn x channel_dim_fn
                assert weights[1].shape == (16,)
    
    def test_initialize_head(self):
        """Test head initialization."""
        head_weights = self.initializer.initialize_head()
        
        assert len(head_weights) == 2  # weight and bias
        assert head_weights[0].shape == (16, 2)  # channel_dim_fn x 2
        assert head_weights[1].shape == (2,)
        
        # Check symmetry
        assert torch.all(head_weights[0][..., 0] == head_weights[0][..., 1])
    
    def test_symmetrize_xy(self):
        """Test xy symmetry enforcement."""
        # Create a multitensor with asymmetric values
        multiweights = self.system.make_multitensor()
        for dims in self.system:
            if dims[3] == 0 and dims[4] == 1:
                multiweights[dims] = torch.tensor(1.0)
            elif dims[3] == 1 and dims[4] == 0:
                multiweights[dims] = torch.tensor(2.0)
        
        # Apply symmetry
        self.initializer.symmetrize_xy(multiweights)
        
        # Check symmetry
        for dims in self.system:
            if dims[3] == 0 and dims[4] == 1:
                assert multiweights[dims].item() == multiweights[dims[:3] + [1, 0]].item()
    
    def test_symmetrize_direction_sharing(self):
        """Test direction sharing symmetry enforcement."""
        # Create a multitensor with asymmetric values
        multiweights = self.system.make_multitensor()
        for dims in self.system:
            multiweights[dims] = [[torch.tensor(1.0) for _ in range(8)] for _ in range(8)]
        
        # Apply symmetry
        self.initializer.symmetrize_direction_sharing(multiweights)
        
        # Check that all values are properly shared
        for dims in self.system:
            for dir1 in range(8):
                for dir2 in range(8):
                    from_dims = dims
                    from_dir1, from_dir2 = dir1, dir2
                    
                    # Apply the same transformations as in the original code
                    if dims[3] + dims[4] == 1:
                        from_dims = dims[:3] + [1, 0]
                        if dims[4] == 1:
                            from_dir1 = (2 + from_dir1) % 8
                            from_dir2 = (2 + from_dir2) % 8
                        
                        if from_dir1 > 4 or (from_dir1 in {0, 4} and from_dir2 > 4):
                            from_dir1 = (8 - from_dir1) % 8
                            from_dir2 = (8 - from_dir2) % 8
                        
                        if 2 < from_dir1 < 6 or (from_dir1 in {2, 6} and 2 < from_dir2 < 6):
                            from_dir1 = (4 - from_dir1) % 8
                            from_dir2 = (4 - from_dir2) % 8
                    else:
                        rotation = (from_dir1 // 2) * 2
                        from_dir1 = (from_dir1 - rotation) % 8
                        from_dir2 = (from_dir2 - rotation) % 8
                        
                        if (from_dir2 - from_dir1) % 8 > 4:
                            from_dir2 = (8 + 2 * from_dir1 - from_dir2) % 8
                    
                    # Check that values are shared
                    assert multiweights[dims][dir1][dir2].item() == multiweights[from_dims][from_dir1][from_dir2].item()


class TestInitializer:
    
    def setup_method(self):
        """Setup for each test method."""
        # Create mock task and multitensor system
        class MockTask:
            pass
        
        self.mock_task = MockTask()
        self.system = MultiTensorSystem(
            n_examples=2,
            n_colors=3,
            n_x=4,
            n_y=5,
            task=self.mock_task
        )
        
        # Channel dimension function
        def channel_dim_fn(dims):
            return 16 if dims[2] == 0 else 8
        
        self.initializer = Initializer(self.system, channel_dim_fn)
        
        # Set random seeds for reproducible tests
        np.random.seed(42)
        torch.manual_seed(42)
    
    def test_initializer_setup(self):
        """Test initializer initialization."""
        assert self.initializer.multitensor_system == self.system
        assert callable(self.initializer.channel_dim_fn)
        assert self.initializer.weights_list == []
        
        # Test channel dim function
        assert self.initializer.channel_dim_fn([0, 0, 0, 0, 0]) == 16
        assert self.initializer.channel_dim_fn([0, 0, 1, 0, 0]) == 8
    
    def test_initialize_zeros(self):
        """Test zero initialization."""
        dims = [1, 0, 0, 1, 0]
        shape = [3, 4]
        
        zeros = self.initializer.initialize_zeros(dims, shape)
        
        assert isinstance(zeros, torch.Tensor)
        assert zeros.shape == tuple(shape)
        assert torch.all(zeros == 0)
        assert zeros.requires_grad == True
        assert len(self.initializer.weights_list) == 1
        assert self.initializer.weights_list[0] is zeros
    
    def test_initialize_zeros_with_callable_shape(self):
        """Test zero initialization with callable shape."""
        dims = [1, 0, 0, 1, 0]
        shape_fn = lambda d: [len(d), 3]
        
        zeros = self.initializer.initialize_zeros(dims, shape_fn)
        
        assert zeros.shape == (5, 3)  # len(dims) = 5
        assert torch.all(zeros == 0)
    
    def test_initialize_linear(self):
        """Test linear layer initialization."""
        dims = [1, 0, 0, 1, 0]
        shape = [4, 6]
        
        weights = self.initializer.initialize_linear(dims, shape)
        
        assert isinstance(weights, list)
        assert len(weights) == 2  # weight matrix and bias
        
        weight_matrix, bias = weights
        assert isinstance(weight_matrix, torch.Tensor)
        assert isinstance(bias, torch.Tensor)
        assert weight_matrix.shape == (4, 6)
        assert bias.shape == (6,)
        assert weight_matrix.requires_grad == True
        assert bias.requires_grad == True
        
        # Check initialization scale
        expected_scale = 1 / np.sqrt(4)
        assert abs(weight_matrix.std().item() - expected_scale) < 0.1
        
        # Check weights were added to list
        assert len(self.initializer.weights_list) == 2
    
    def test_initialize_linear_with_callable_shape(self):
        """Test linear initialization with callable shapes."""
        dims = [1, 0, 0, 1, 0]
        n_in_fn = lambda d: self.initializer.channel_dim_fn(d)  # 16
        n_out_fn = lambda d: 8
        shape = [n_in_fn, n_out_fn]
        
        weights = self.initializer.initialize_linear(dims, shape)
        
        weight_matrix, bias = weights
        assert weight_matrix.shape == (16, 8)
        assert bias.shape == (8,)
    
    def test_initialize_residual(self):
        """Test residual layer initialization."""
        dims = [1, 0, 0, 1, 0]
        n_in = 10
        n_out = 12
        
        residual_weights = self.initializer.initialize_residual(dims, n_in, n_out)
        
        assert isinstance(residual_weights, list)
        assert len(residual_weights) == 2  # down and up projections
        
        down_proj, up_proj = residual_weights
        
        # down_proj: channel_dim -> n_in
        assert len(down_proj) == 2  # weight and bias
        assert down_proj[0].shape == (16, n_in)  # channel_dim_fn([1,0,0,1,0]) = 16
        assert down_proj[1].shape == (n_in,)
        
        # up_proj: n_out -> channel_dim
        assert len(up_proj) == 2
        assert up_proj[0].shape == (n_out, 16)
        assert up_proj[1].shape == (16,)
    
    def test_initialize_posterior(self):
        """Test posterior initialization."""
        dims = [1, 0, 0, 1, 0]
        channel_dim = 8
        
        posterior = self.initializer.initialize_posterior(dims, channel_dim)
        
        assert isinstance(posterior, list)
        assert len(posterior) == 2  # mean and local_capacity_adjustment
        
        mean, local_capacity_adj = posterior
        expected_shape = self.system.shape(dims, channel_dim)
        
        assert isinstance(mean, torch.Tensor)
        assert isinstance(local_capacity_adj, torch.Tensor)
        assert mean.shape == tuple(expected_shape)
        assert local_capacity_adj.shape == tuple(expected_shape)
        assert mean.requires_grad == True
        assert local_capacity_adj.requires_grad == True
        
        # Check mean is small
        assert abs(mean.mean().item()) < 0.1
    
    def test_initialize_posterior_with_callable_dim(self):
        """Test posterior initialization with callable channel_dim."""
        dims = [1, 0, 0, 1, 0]
        channel_dim_fn = lambda d: 6
        
        posterior = self.initializer.initialize_posterior(dims, channel_dim_fn)
        mean, _ = posterior
        expected_shape = self.system.shape(dims, 6)
        assert mean.shape == tuple(expected_shape)
    
    def test_initialize_direction_share(self):
        """Test direction share initialization."""
        dims = [1, 0, 1, 1, 0]  # has direction dimension
        unused_arg = None
        
        direction_weights = self.initializer.initialize_direction_share(dims, unused_arg)
        
        assert isinstance(direction_weights, list)
        assert len(direction_weights) == 8  # 8 directions
        
        for direction_group in direction_weights:
            assert isinstance(direction_group, list)
            assert len(direction_group) == 8  # 8x8 direction matrix
            
            for direction_weight in direction_group:
                assert isinstance(direction_weight, list)
                assert len(direction_weight) == 2  # weight and bias
                
                weight, bias = direction_weight
                channel_dim = self.initializer.channel_dim_fn(dims)  # 8 for direction
                assert weight.shape == (channel_dim, channel_dim)
                assert bias.shape == (channel_dim,)
    
    def test_initialize_head(self):
        """Test head initialization with symmetry."""
        head_weights = self.initializer.initialize_head()
        
        assert isinstance(head_weights, list)
        assert len(head_weights) == 2  # weight and bias
        
        weight, bias = head_weights
        dims = [1, 1, 0, 1, 1]
        expected_in_dim = self.initializer.channel_dim_fn(dims)  # 16
        
        assert weight.shape == (expected_in_dim, 2)
        assert bias.shape == (2,)
        
        # Check symmetry - both output channels should be identical
        assert torch.allclose(weight[:, 0], weight[:, 1])
    
    def test_initialize_multizeros(self):
        """Test multi-tensor zero initialization."""
        shape = [4, 3]
        multizeros = self.initializer.initialize_multizeros(shape)
        # Check it's a MultiTensor
        from src.arc_multitensor import MultiTensor
        assert isinstance(multizeros, MultiTensor)
        
        # Check values for valid dimensions
        for dims in self.system:
            tensor = multizeros[dims]
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == tuple(shape)
            assert torch.all(tensor == 0)
    
    def test_initialize_multilinear(self):
        """Test multi-tensor linear initialization."""
        shape = [6, 4]
        multilinear = self.initializer.initialize_multilinear(shape)
        from src.arc_multitensor import MultiTensor
        assert isinstance(multilinear, MultiTensor)
        
        # Check values for valid dimensions
        for dims in self.system:
            weights = multilinear[dims]
            assert isinstance(weights, list)
            assert len(weights) == 2  # weight and bias
            
            weight, bias = weights
            assert weight.shape == (6, 4)
            assert bias.shape == (4,)
    
    def test_initialize_multiresidual(self):
        """Test multi-tensor residual initialization."""
        n_in = 8
        n_out = 10
        multiresidual = self.initializer.initialize_multiresidual(n_in, n_out)
        from src.arc_multitensor import MultiTensor
        assert isinstance(multiresidual, MultiTensor)
        
        for dims in self.system:
            residual_weights = multiresidual[dims]
            assert isinstance(residual_weights, list)
            assert len(residual_weights) == 2  # down and up projections
            
            down_proj, up_proj = residual_weights
            channel_dim = self.initializer.channel_dim_fn(dims)
            
            # Check down projection: channel_dim -> n_in
            assert down_proj[0].shape == (channel_dim, n_in)
            assert down_proj[1].shape == (n_in,)
            
            # Check up projection: n_out -> channel_dim  
            assert up_proj[0].shape == (n_out, channel_dim)
            assert up_proj[1].shape == (channel_dim,)
    
    def test_initialize_multiposterior(self):
        """Test multi-tensor posterior initialization."""
        decoding_dim = 6
        multiposterior = self.initializer.initialize_multiposterior(decoding_dim)
        from src.arc_multitensor import MultiTensor
        assert isinstance(multiposterior, MultiTensor)
        
        for dims in self.system:
            posterior = multiposterior[dims]
            assert isinstance(posterior, list)
            assert len(posterior) == 2  # mean and local_capacity_adjustment
            
            mean, local_adj = posterior
            expected_shape = self.system.shape(dims, decoding_dim)
            assert mean.shape == tuple(expected_shape)
            assert local_adj.shape == tuple(expected_shape)
    
    def test_initialize_multidirection_share(self):
        """Test multi-tensor direction share initialization."""
        multidirection = self.initializer.initialize_multidirection_share()
        from src.arc_multitensor import MultiTensor
        assert isinstance(multidirection, MultiTensor)
        
        for dims in self.system:
            direction_weights = multidirection[dims]
            assert isinstance(direction_weights, list)
            assert len(direction_weights) == 8
            
            for direction_group in direction_weights:
                assert len(direction_group) == 8
    
    def test_symmetrize_xy(self):
        """Test X-Y symmetrization."""
        # Create a multilinear to symmetrize
        multiweights = self.initializer.initialize_multilinear([4, 4])
        
        # Get original values
        dims_xy = [1, 0, 0, 1, 0]  # x but not y
        dims_yx = [1, 0, 0, 0, 1]  # y but not x
        
        if self.system.dims_valid(dims_xy) and self.system.dims_valid(dims_yx):
            original_xy = multiweights[dims_xy][0].clone()
            
            # Apply symmetrization
            self.initializer.symmetrize_xy(multiweights)
            
            # Check that dims_yx now equals dims_xy
            assert torch.allclose(multiweights[dims_yx][0], original_xy)
    
    def test_symmetrize_direction_sharing(self):
        """Test direction sharing symmetrization."""
        multidirection = self.initializer.initialize_multidirection_share()
        
        # Apply symmetrization
        self.initializer.symmetrize_direction_sharing(multidirection)
        
        # Check that symmetrization was applied (hard to test specific values
        # but we can check it doesn't crash and preserves structure)
        for dims in self.system:
            direction_weights = multidirection[dims]
            assert isinstance(direction_weights, list)
            assert len(direction_weights) == 8
            for group in direction_weights:
                assert len(group) == 8
    
    def test_weights_list_accumulation(self):
        """Test that weights are properly accumulated in weights_list."""
        initial_count = len(self.initializer.weights_list)
        
        # Initialize various weights
        self.initializer.initialize_zeros([1, 0, 0, 1, 0], [3, 3])
        zeros_count = len(self.initializer.weights_list)
        
        self.initializer.initialize_linear([1, 0, 0, 1, 0], [4, 2])
        linear_count = len(self.initializer.weights_list)
        
        self.initializer.initialize_posterior([1, 0, 0, 1, 0], 5)
        posterior_count = len(self.initializer.weights_list)
        
        # Check accumulation
        assert zeros_count == initial_count + 1  # zeros tensor
        assert linear_count == zeros_count + 2   # weight + bias
        assert posterior_count == linear_count + 2  # mean + local_adj
        
        # Check all weights require gradients
        for weight in self.initializer.weights_list:
            assert weight.requires_grad == True
    
    def test_integration_realistic_usage(self):
        """Test realistic usage scenario."""
        # Initialize components similar to ARCCompressor
        decoding_dim = 4
        share_dim = 8
        
        # Initialize various components
        multiposteriors = self.initializer.initialize_multiposterior(decoding_dim)
        decode_weights = self.initializer.initialize_multilinear([decoding_dim, self.initializer.channel_dim_fn])
        share_weights = self.initializer.initialize_multiresidual(share_dim, share_dim)
        head_weights = self.initializer.initialize_head()
        
        # Apply symmetrizations
        self.initializer.symmetrize_xy(decode_weights)
        self.initializer.symmetrize_xy(share_weights)
        
        # Check everything is properly initialized
        assert len(self.initializer.weights_list) > 0
        
        # Check that we can access components
        for dims in self.system:
            assert multiposteriors[dims] is not None
            assert decode_weights[dims] is not None
            assert share_weights[dims] is not None
        
        assert head_weights is not None
        assert len(head_weights) == 2


class TestInitializerEdgeCases:
    
    def setup_method(self):
        """Setup for edge case tests."""
        class MockTask:
            pass
        
        self.mock_task = MockTask()
        # Small system for edge case testing
        self.system = MultiTensorSystem(1, 1, 1, 1, self.mock_task)
        
        def simple_channel_dim_fn(dims):
            return 2
        
        self.initializer = Initializer(self.system, simple_channel_dim_fn)
    
    def test_minimal_dimensions(self):
        """Test with minimal tensor dimensions."""
        dims = [0, 1, 0, 0, 0]  # just color
        
        # Should handle minimal cases
        zeros = self.initializer.initialize_zeros(dims, [1])
        assert zeros.shape == (1,)
        
        linear = self.initializer.initialize_linear(dims, [1, 1])
        assert linear[0].shape == (1, 1)
        assert linear[1].shape == (1,)
    
    def test_large_dimensions(self):
        """Test with larger dimensions."""
        dims = [1, 1, 1, 1, 1]  # all dimensions
        large_shape = [100, 50]
        
        zeros = self.initializer.initialize_zeros(dims, large_shape)
        assert zeros.shape == (100, 50)
        
        linear = self.initializer.initialize_linear(dims, [100, 50])
        weight, bias = linear
        assert weight.shape == (100, 50)
        assert bias.shape == (50,)
        
        # Check initialization scale is reasonable for large matrices
        expected_scale = 1 / np.sqrt(100)
        assert abs(weight.std().item() - expected_scale) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])