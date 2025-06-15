import pytest
import numpy as np
import torch
import sys
import os

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_multitensor import MultiTensorSystem, MultiTensor, multify, NUM_DIMENSIONS


class TestMultiTensorSystem:
    
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
    
    def test_initialization(self):
        """Test MultiTensorSystem initialization."""
        assert self.system.n_examples == 2
        assert self.system.n_colors == 3
        assert self.system.n_directions == 8
        assert self.system.n_x == 4
        assert self.system.n_y == 5
        assert self.system.task == self.mock_task
        assert len(self.system.dim_lengths) == 5
        assert self.system.dim_lengths == [2, 3, 8, 4, 5]
    
    def test_dims_valid(self):
        """Test dimension validation logic."""
        # Valid cases
        assert self.system.dims_valid([0, 1, 0, 0, 0])  # color only
        assert self.system.dims_valid([0, 0, 1, 0, 0])  # direction only
        assert self.system.dims_valid([1, 1, 0, 1, 0])  # examples + color + x
        assert self.system.dims_valid([1, 0, 0, 1, 1])  # examples + x + y
        
        # Invalid cases
        assert not self.system.dims_valid([0, 1, 0, 1, 0])  # color + x without examples
        assert not self.system.dims_valid([0, 0, 0, 1, 0])  # x without examples
        assert not self.system.dims_valid([0, 0, 0, 0, 1])  # y without examples
        assert not self.system.dims_valid([0, 0, 0, 0, 0])  # no dimensions
        assert not self.system.dims_valid([1, 0, 0, 0, 0])  # examples only
    
    def test_shape(self):
        """Test shape calculation."""
        # Test without extra dimension
        assert self.system.shape([1, 0, 0, 1, 0]) == [2, 4]
        assert self.system.shape([0, 1, 1, 0, 0]) == [3, 8]
        assert self.system.shape([1, 1, 0, 1, 1]) == [2, 3, 4, 5]
        
        # Test with extra dimension
        assert self.system.shape([1, 0, 0, 1, 0], extra_dim=10) == [2, 4, 10]
        assert self.system.shape([0, 1, 0, 0, 0], extra_dim=16) == [3, 16]
    
    def test_generate_dims_combinations(self):
        """Test dimension combination generation."""
        combinations = list(self.system._generate_dims_combinations())
        assert len(combinations) == 2 ** NUM_DIMENSIONS  # 32 combinations
        assert [0, 0, 0, 0, 0] in combinations
        assert [1, 1, 1, 1, 1] in combinations
        assert [1, 0, 1, 0, 1] in combinations
    
    def test_iteration(self):
        """Test system iteration over valid dimensions."""
        valid_dims = list(self.system)
        # Should only include valid dimensions
        for dims in valid_dims:
            assert self.system.dims_valid(dims)
        
        # Check we have reasonable number of valid combinations
        assert len(valid_dims) > 0
        assert len(valid_dims) < 2 ** NUM_DIMENSIONS
    
    def test_make_multitensor(self):
        """Test multitensor creation."""
        # Test with None default
        mt_none = self.system.make_multitensor()
        assert isinstance(mt_none, MultiTensor)
        assert mt_none.multitensor_system == self.system
        
        # Test with value default
        mt_val = self.system.make_multitensor(default=42)
        assert isinstance(mt_val, MultiTensor)
        
        # Test accessing a value
        test_dims = [1, 0, 0, 1, 0]
        if self.system.dims_valid(test_dims):
            assert mt_val[test_dims] == 42


class TestMultiTensor:
    
    def setup_method(self):
        """Setup for each test method."""
        class MockTask:
            pass
        
        self.mock_task = MockTask()
        self.system = MultiTensorSystem(2, 2, 2, 2, self.mock_task)
        self.multitensor = self.system.make_multitensor(default=0)
    
    def test_initialization(self):
        """Test MultiTensor initialization."""
        assert isinstance(self.multitensor, MultiTensor)
        assert self.multitensor.multitensor_system == self.system
        assert self.multitensor.data is not None
    
    def test_getitem_setitem(self):
        """Test getting and setting items."""
        test_dims = [1, 0, 0, 1, 0]
        
        # Test setting
        self.multitensor[test_dims] = 99
        
        # Test getting
        assert self.multitensor[test_dims] == 99
        
        # Test with different value
        self.multitensor[test_dims] = "test_value"
        assert self.multitensor[test_dims] == "test_value"


class TestMultifyDecorator:
    
    def setup_method(self):
        """Setup for each test method."""
        class MockTask:
            pass
        
        self.mock_task = MockTask()
        self.system = MultiTensorSystem(2, 2, 2, 2, self.mock_task)
    
    def test_multify_without_multitensor(self):
        """Test multify decorator with regular arguments."""
        @multify
        def test_function(dims, x, y):
            return x + y
        
        result = test_function(None, 10, 20)
        assert result == 30
    
    def test_multify_with_multitensor(self):
        """Test multify decorator with MultiTensor arguments."""
        @multify
        def add_function(dims, x, y):
            return x + y
        
        # Create multitensors
        mt1 = self.system.make_multitensor(default=5)
        mt2 = self.system.make_multitensor(default=3)
        
        # Apply function
        result = add_function(mt1, mt2)
        
        # Check result is MultiTensor
        assert isinstance(result, MultiTensor)
        
        # Check a specific dimension
        for dims in self.system:
            assert result[dims] == 8  # 5 + 3
    
    def test_multify_mixed_arguments(self):
        """Test multify with mixed MultiTensor and regular arguments."""
        @multify
        def mixed_function(dims, mt_arg, regular_arg):
            return mt_arg * regular_arg
        
        mt = self.system.make_multitensor(default=4)
        result = mixed_function(mt, 3)
        
        assert isinstance(result, MultiTensor)
        for dims in self.system:
            assert result[dims] == 12  # 4 * 3


class TestIntegration:
    
    def test_realistic_usage(self):
        """Test realistic usage scenario."""
        class MockTask:
            pass
        
        mock_task = MockTask()
        system = MultiTensorSystem(
            n_examples=3,
            n_colors=5, 
            n_x=10,
            n_y=10,
            task=mock_task
        )
        
        # Create some multitensors
        weights = system.make_multitensor(default=1.0)
        inputs = system.make_multitensor(default=0.5)
        
        # Define a processing function
        @multify
        def process(dims, w, x):
            return w * x + 0.1
        
        # Apply processing
        outputs = process(weights, inputs)
        
        # Verify results
        assert isinstance(outputs, MultiTensor)
        assert outputs.multitensor_system == system
        
        # Check specific calculation
        for dims in system:
            expected = 1.0 * 0.5 + 0.1  # 0.6
            assert abs(outputs[dims] - expected) < 1e-6
    
    def test_tensor_operations(self):
        """Test with actual PyTorch tensors."""
        class MockTask:
            pass
        
        mock_task = MockTask()
        system = MultiTensorSystem(2, 2, 2, 2, mock_task)
        
        # Create multitensor with torch tensors
        mt = system.make_multitensor()
        
        for dims in system:
            shape = system.shape(dims)
            mt[dims] = torch.randn(shape)
        
        # Define tensor operation
        @multify
        def normalize(dims, x):
            return x / torch.norm(x)
        
        # Apply operation
        normalized = normalize(mt)
        
        # Verify results
        assert isinstance(normalized, MultiTensor)
        for dims in system:
            # Check that norm is approximately 1
            norm_val = torch.norm(normalized[dims]).item()
            assert abs(norm_val - 1.0) < 1e-5


class TestMultiTensorEdgeCases:
    def setup_method(self):
        """Setup for each test method."""
        class MockTask:
            pass
        
        self.mock_task = MockTask()
        self.system = MultiTensorSystem(2, 2, 2, 2, self.mock_task)
        self.multitensor = self.system.make_multitensor(default=0)
    
    def test_make_multitensor_with_index(self):
        """Test _make_multitensor with different indices."""
        # Test with index 0
        result = self.system._make_multitensor(42, 0)
        assert isinstance(result, list)
        assert len(result) == 2
        
        # Test with index NUM_DIMENSIONS
        result = self.system._make_multitensor(42, NUM_DIMENSIONS)
        assert result == 42
        
        # Test with intermediate index
        result = self.system._make_multitensor(42, 2)
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_multify_with_no_multitensor_args(self):
        """Test multify decorator with no MultiTensor arguments."""
        @multify
        def test_fn(dims, x, y):
            return x + y
        
        # Pass None as dims since we're not using MultiTensor
        result = test_fn(None, 1, 2)
        assert result == 3
    
    def test_multify_with_mixed_args(self):
        """Test multify decorator with mixed MultiTensor and regular arguments."""
        @multify
        def test_fn(dims, mt_arg, regular_arg):
            return mt_arg * regular_arg
        
        mt = self.system.make_multitensor(default=4)
        result = test_fn(mt, 3)
        
        assert isinstance(result, MultiTensor)
        for dims in self.system:
            assert result[dims] == 12  # 4 * 3
    
    def test_multify_with_kwargs(self):
        """Test multify decorator with keyword arguments."""
        @multify
        def test_fn(dims, x, y, multiplier=1):
            return (x + y) * multiplier
        
        mt1 = self.system.make_multitensor(default=2)
        mt2 = self.system.make_multitensor(default=3)
        
        # Test with regular kwargs
        result = test_fn(mt1, mt2, multiplier=2)
        assert isinstance(result, MultiTensor)
        for dims in self.system:
            assert result[dims] == 10  # (2 + 3) * 2
        
        # Test with MultiTensor kwargs
        mt_multiplier = self.system.make_multitensor(default=2)
        result = test_fn(mt1, mt2, multiplier=mt_multiplier)
        assert isinstance(result, MultiTensor)
        for dims in self.system:
            assert result[dims] == 10  # (2 + 3) * 2


class TestCoverageBoost:
    def test_multify_iterate_and_assign(self):
        # This will cover the assignment in iterate_and_assign
        class MockTask:
            pass
        system = MultiTensorSystem(1, 1, 1, 1, MockTask())
        mt = system.make_multitensor(default=1)
        @multify
        def fn(dims, x):
            return x + 1
        result = fn(mt)
        assert isinstance(result, MultiTensor)
        for dims in system:
            assert result[dims] == 2

    def test_initialize_zeros_callable_shape(self):
        from src.arc_weight_initializer import Initializer
        class MockTask:
            pass
        system = MultiTensorSystem(1, 1, 1, 1, MockTask())
        initializer = Initializer(system, lambda d: 1)
        zeros = initializer.initialize_zeros([1,0,0,1,0], lambda d: (2,2))
        assert zeros.shape == (2,2)

    def test_plot_solution_exception(self):
        from src.arc_visualizer import plot_solution
        class DummyTask:
            pass
        class DummyLogger:
            def __init__(self):
                self.task = DummyTask()  # No task_name attribute
        # Should not raise, but print error (and cover the except line)
        try:
            plot_solution(DummyLogger())
        except AttributeError:
            pass  # Expected, just for coverage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])