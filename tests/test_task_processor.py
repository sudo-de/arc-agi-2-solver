import pytest
import json
import os
import torch
import numpy as np
from src.arc_task_processor import ARCTask, load_and_process_tasks

@pytest.fixture
def sample_task():
    """Load a sample ARC-AGI task from JSON."""
    json_path = os.path.join(os.path.dirname(__file__), 'data/sample_task.json')
    with open(json_path, 'r') as f:
        problems = json.load(f)
    task_data = problems['00576224']
    solution = task_data['test'][0]['output']  # Extract test output as solution
    task = ARCTask('00576224', task_data, [solution])
    return task

@pytest.fixture
def device():
    """Set default device for tests."""
    torch.set_default_dtype(torch.float32)
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(autouse=True)
def setup_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(0)
    np.random.seed(0)

def test_task_initialization(sample_task):
    """Test ARCTask initialization."""
    assert sample_task.task_id == '00576224'
    assert sample_task.n_train == 1
    assert sample_task.n_test == 1
    assert sample_task.n_examples == 2
    assert sample_task.max_x == 2
    assert sample_task.max_y == 2
    assert sample_task.n_colors == 1  # Colors: [0, 1] -> n_colors = len(colors) - 1 = 1
    assert sample_task.input_output_same_size
    assert sample_task.all_inputs_same_size
    assert sample_task.all_outputs_same_size
    expected_solution = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
    assert torch.equal(sample_task.solution_tensor, expected_solution)

def test_problem_tensor_shape(sample_task):
    """Test problem_tensor shape."""
    assert sample_task.problem_tensor.shape == (2, 2, 2, 2)  # [examples, x, y, mode]

def test_load_and_process_tasks(monkeypatch):
    """Test task loading function with mocked file loading."""
    mock_problems = {
        '00576224': {
            'train': [{'input': [[0, 1], [1, 0]], 'output': [[0, 0], [1, 1]]}],
            'test': [{'input': [[1, 0], [0, 1]], 'output': [[0, 0], [1, 1]]}]
        }
    }
    def mock_open(*args, **kwargs):
        class MockFile:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps(mock_problems)
        return MockFile()
    monkeypatch.setattr('builtins.open', mock_open)
    tasks = load_and_process_tasks('test', ['00576224'])
    assert len(tasks) == 1
    assert tasks[0].task_id == '00576224'

def test_invalid_task_id(monkeypatch):
    """Test loading non-existent task."""
    def mock_open(*args, **kwargs):
        class MockFile:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({})
        return MockFile()
    monkeypatch.setattr('builtins.open', mock_open)
    with pytest.raises(ValueError, match=r"Some tasks in \['invalid_id'\] were not found in test split."):
        load_and_process_tasks('test', ['invalid_id'])