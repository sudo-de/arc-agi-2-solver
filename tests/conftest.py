"""
Pytest configuration and fixtures for ARC-AGI testing.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment for all tests."""
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use CPU for testing by default
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    
    # Disable gradients for faster testing (enable per test if needed)
    torch.set_grad_enabled(True)
    
    yield
    
    # Cleanup after all tests
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_download_functions():
    """Mock all download functions to avoid network dependencies."""
    with patch('scripts.download_data.download_from_url', return_value=True) as mock_url, \
         patch('scripts.download_data.download_kaggle_dataset', return_value=True) as mock_kaggle, \
         patch('scripts.download_data.download_from_github', return_value=True) as mock_github:
        yield {
            'url': mock_url,
            'kaggle': mock_kaggle,
            'github': mock_github
        }


@pytest.fixture
def sample_arc_problem():
    """Create a sample ARC problem for testing."""
    return {
        'train': [
            {
                'input': [[0, 1], [1, 0]],
                'output': [[1, 0], [0, 1]]
            },
            {
                'input': [[2, 3], [3, 2]],
                'output': [[3, 2], [2, 3]]
            }
        ],
        'test': [
            {
                'input': [[1, 2], [2, 1]]
            }
        ]
    }


@pytest.fixture
def sample_arc_solution():
    """Create a sample ARC solution for testing."""
    return [[[2, 1], [1, 2]]]


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield data_dir
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)


@pytest.fixture
def large_arc_problem():
    """Provide a larger ARC problem for testing."""
    large_grid = [[i % 10 for i in range(10)] for _ in range(10)]
    return {
        'train': [
            {
                'input': large_grid[:5],
                'output': large_grid[:5]
            }
        ],
        'test': [
            {
                'input': large_grid[:3]
            }
        ]
    }


@pytest.fixture
def minimal_arc_problem():
    """Provide a minimal 1x1 ARC problem for edge case testing."""
    return {
        'train': [
            {
                'input': [[5]],
                'output': [[7]]
            }
        ],
        'test': [
            {
                'input': [[5]]
            }
        ]
    }


@pytest.fixture
def mock_task_with_multitensor():
    """Create a mock task with multitensor system for testing."""
    from src.arc_task_processor import Task
    
    problem = {
        'train': [
            {
                'input': [[0, 1], [1, 0]],
                'output': [[1, 0], [0, 1]]
            }
        ],
        'test': [
            {
                'input': [[1, 2], [2, 1]]
            }
        ]
    }
    
    return Task("mock_task", problem, None)


@pytest.fixture
def torch_device():
    """Provide appropriate torch device for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "cpu: mark test as CPU only")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test characteristics."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "large" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or "realistic" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (most tests)
        if not any(marker in item.name for marker in ["integration", "slow", "performance"]):
            item.add_marker(pytest.mark.unit)
        
        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        else:
            item.add_marker(pytest.mark.cpu)


@pytest.fixture
def memory_tracker():
    """Track memory usage during test execution."""
    import psutil
    import gc
    
    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    # Cleanup and check final memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Warn if memory increased significantly
    if memory_increase > 100:  # More than 100MB increase
        pytest.warns(UserWarning, f"Memory increased by {memory_increase:.1f}MB during test")


# Skip tests requiring GPU if not available
def pytest_runtest_setup(item):
    """Setup function run before each test."""
    if "gpu" in [mark.name for mark in item.iter_markers()]:
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")


# Custom assertions
def assert_tensor_shape(tensor, expected_shape):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_finite(tensor):
    """Assert all tensor values are finite."""
    assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"


def assert_tensor_range(tensor, min_val=None, max_val=None):
    """Assert tensor values are within expected range."""
    if min_val is not None:
        assert tensor.min() >= min_val, f"Tensor minimum {tensor.min()} below {min_val}"
    if max_val is not None:
        assert tensor.max() <= max_val, f"Tensor maximum {tensor.max()} above {max_val}"


# Add custom assertions to pytest namespace
pytest.assert_tensor_shape = assert_tensor_shape
pytest.assert_tensor_finite = assert_tensor_finite
pytest.assert_tensor_range = assert_tensor_range


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    return {
        "task1": {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]]
                }
            ],
            "test": [
                {
                    "input": [[1, 2], [2, 1]]
                }
            ]
        }
    }