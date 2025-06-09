import pytest
import torch
import numpy as np
from src.arc_multitensor import ARCMultiTensorSystem, ARCMultiTensor, apply_multitensor

@pytest.fixture
def sample_task():
    class DummyTask:
        n_examples = 2
        n_colors = 3
        max_x = 4
        max_y = 5
    return DummyTask()

def test_multitensor_system(sample_task):
    """Test ARCMultiTensorSystem initialization and iteration."""
    mts = ARCMultiTensorSystem(
        n_examples=sample_task.n_examples,
        n_colors=sample_task.n_colors,
        n_x=sample_task.max_x,
        n_y=sample_task.max_y,
        task=sample_task
    )
    dims_list = list(mts)
    assert len(dims_list) > 0
    assert all(len(dims) == 5 for dims in dims_list)
    assert all(all(d in (0, 1) for d in dims) for dims in dims_list)

def test_multitensor_creation(sample_task):
    """Test ARCMultiTensor creation and access."""
    mts = ARCMultiTensorSystem(
        n_examples=sample_task.n_examples,
        n_colors=sample_task.n_colors,
        n_x=sample_task.max_x,
        n_y=sample_task.max_y,
        task=sample_task
    )
    mt = mts.create_multitensor(default=torch.zeros(2, 2))
    assert isinstance(mt, ARCMultiTensor)
    for dims in mts:
        assert mt[dims].shape == (2, 2)

def test_apply_multitensor(sample_task):
    """Test apply_multitensor function."""
    mts = ARCMultiTensorSystem(
        n_examples=sample_task.n_examples,
        n_colors=sample_task.n_colors,
        n_x=sample_task.max_x,
        n_y=sample_task.max_y,
        task=sample_task
    )
    def init_fn(dims, tensor, value=1.0):
        shape = tuple(tensor.shape)
        return torch.full(shape, value, dtype=torch.float32)
    input_mt = mts.create_multitensor(default=torch.zeros(3, 3))
    result_mt = apply_multitensor(init_fn)(None, input_mt, value=2.0)
    assert isinstance(result_mt, ARCMultiTensor)
    for dims in mts:
        assert result_mt[dims].shape == (3, 3)
        assert torch.all(result_mt[dims] == 2.0)