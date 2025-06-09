import pytest
import torch
from src.arc_multitensor import ARCMultiTensor
from src.arc_network_layers import decode_latents, share_up, softmax, normalize, direction_share
from src.arc_task_processor import ARCTask
import os
import json

@pytest.fixture
def sample_task():
    """Load a sample ARC-AGI task from JSON."""
    json_path = os.path.join(os.path.dirname(__file__), 'data/sample_task.json')
    with open(json_path, 'r') as f:
        problems = json.load(f)
    task_data = problems['00576224']
    solution = task_data['test'][0]['output']
    task = ARCTask('00576224', task_data, [solution])
    return task

def test_decode_latents(sample_task):
    """Test decode_latents function."""
    mts = sample_task.multitensor_system
    target_caps = mts.create_multitensor(default=torch.tensor(1.0))
    weight_matrix = torch.randn(2, 2)  # Input channels=2, output channels=2
    bias = torch.zeros(2)
    decode_weights = mts.create_multitensor(default=[weight_matrix, bias])
    mean = torch.randn(2, 2, 2, 2)  # Shape: (n examples=2, max_x=2, max_y=2, channels=2)
    adjustment = torch.zeros(2, 2, 2, 2)
    posteriors = mts.create_multitensor(default=(mean, adjustment))
    
    decoded, kl_divergences, kl_names = decode_latents(target_caps, decode_weights, posteriors)
    assert isinstance(decoded, ARCMultiTensor), f"Expected ARCMultiTensor, got {type(decoded)}"
    for dims in mts:
        assert decoded[dims].shape == (2, 2, 2, 2), f"Expected shape (2, 2, 2, 2) for dims {dims}, got {decoded[dims].shape}"
    assert len(kl_divergences) == len(list(mts)), f"Expected {len(list(mts))} KL divergences, got {len(kl_divergences)}"
    assert len(kl_names) == len(kl_divergences)

def test_share_up(sample_task):
    """Test share_up function."""
    mts = sample_task.multitensor_system
    init = mts.create_multitensor(default=torch.randn(2, 2, 2, 2))
    down_weight = torch.randn(2, 2)  # Channels: 2 -> 2
    up_weight = torch.randn(2, 2)    # Channels: 2 -> 2
    weights = mts.create_multitensor(default=[down_weight, up_weight])
    
    result = share_up(init, weights)
    assert isinstance(result, ARCMultiTensor), f"Expected ARCMultiTensor, got {type(result)}"
    for dims in mts:
        assert result[dims].shape == (2, 2, 2, 2), f"Expected shape (2, 2, 2, 2) for dims {dims}, got {result[dims].shape}"

def test_softmax(sample_task):
    """Test softmax function."""
    mts = sample_task.multitensor_system
    init = mts.create_multitensor(default=torch.randn(2, 2, 2, 2))
    weight1 = torch.randn(2, 4)  # Input channels=2, output channels=4
    bias1 = torch.zeros(4)
    weight2 = torch.randn(4, 2)  # Input channels=4, output channels=2 to match input
    bias2 = torch.zeros(2)
    projection_weights = mts.create_multitensor(default=[[weight1, bias1], [weight2, bias2]])
    
    result = softmax(init, projection_weights=projection_weights, pre_norm=True)
    assert isinstance(result, ARCMultiTensor), f"Expected ARCMultiTensor, got {type(result)}"
    for dims in mts:
        assert result[dims].shape == (2, 2, 2, 2), f"Expected shape (2, 2, 2, 2) for dims {dims}, got {result[dims].shape}"

def test_normalize(sample_task):
    """Test normalize function."""
    init = torch.randn(2, 2, 2, 2)  # Direct tensor, not ARCMultiTensor
    result = normalize(init)
    assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
    assert result.shape == (2, 2, 2, 2), f"Expected shape (2, 2, 2, 2), got {result.shape}"
    variance = torch.mean(result[..., 0]**2, dim=[0, 1, 2])
    assert torch.allclose(variance, torch.tensor(1.0), atol=1e-4), f"Expected unit variance, got {variance}"

def test_direction_share(sample_task):
    """Test direction_share function."""
    mts = sample_task.multitensor_system
    init = mts.create_multitensor(default=torch.randn(2, 2, 8, 2, 2, 2))  # Include direction dimension
    weights = [[[torch.randn(2, 2) for _ in range(8)] for _ in range(8)]]
    weights_mt = mts.create_multitensor(default=weights)
    
    result = direction_share(init, weights, pre_norm=True)
    assert hasattr(result, 'system'), f"Expected ARCMultiTensor, got {type(result)}"
    for dims in mts:
        assert result[dims].shape == (2, 2, 8, 2, 2, 2), f"Expected shape (2, 2, 8, 2, 2, 2) for dims {dims}, got {result[dims].shape}"