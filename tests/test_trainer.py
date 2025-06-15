import pytest
import torch
import numpy as np
from src.arc_trainer import mask_select_logprobs, take_step
from src.arc_task_processor import Task

class TestTrainer:
    def setup_method(self):
        """Setup test environment before each test"""
        # Create a mock model
        class MockModel:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.randn(1))
                self.parameters = lambda: [self.weight]
                def forward():
                    logits = self.weight * torch.randn(2, 3, 4, 5, 2)
                    return (
                        logits,                      # logits: [example, color, x, y, in_out]
                        torch.ones(2, 4, 2),         # x_mask: [example, x, in_out]
                        torch.ones(2, 5, 2),         # y_mask: [example, y, in_out]
                        [torch.tensor(0.1)],         # KL_amounts
                        ["kl_loss"]                 # KL_names
                    )
                self.forward = forward

        # Create a mock logger
        class MockLogger:
            def __init__(self):
                self.logged_data = []
            def log(self, *args, **kwargs):
                self.logged_data.append((args, kwargs))

        self.model = MockModel()
        self.logger = MockLogger()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def test_mask_select_logprobs(self):
        """Test mask selection for log probabilities"""
        # Create test data
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
        length = 2

        # Test the function
        log_partition, logprobs = mask_select_logprobs(mask, length)
        assert isinstance(log_partition, torch.Tensor)
        assert isinstance(logprobs, torch.Tensor)

    def test_take_step(self):
        """Test taking a training step"""
        class MockTask:
            def __init__(self):
                self.problem = torch.randint(0, 3, (2, 4, 5, 2), dtype=torch.long)
                self.solution = torch.randint(0, 3, (2, 4, 5), dtype=torch.long)
                self.masks = torch.ones((2, 4, 5, 2))
                self.n_examples = 2
                self.n_train = 1
                self.n_test = 1
                self.shapes = [[[4, 5], [4, 5]], [[4, 5], [4, 5]]]
                self.in_out_same_size = True
                self.all_out_same_size = False
                self.all_in_same_size = False
        task = MockTask()
        train_step = 1
        take_step(task, self.model, self.optimizer, train_step, self.logger)

    def test_take_step_with_gradient_flow(self):
        """Test gradient flow during training step"""
        class TestModel:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.randn(1))
                def forward():
                    logits = self.weight * torch.randn(2, 3, 4, 5, 2)
                    return (
                        logits,
                        torch.ones(2, 4, 2),
                        torch.ones(2, 5, 2),
                        [torch.tensor(0.1)],
                        ["kl_loss"]
                    )
                self.forward = forward
            def parameters(self):
                return [self.weight]
        model = TestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        class MockTask:
            def __init__(self):
                self.problem = torch.randint(0, 3, (2, 4, 5, 2), dtype=torch.long)
                self.solution = torch.randint(0, 3, (2, 4, 5), dtype=torch.long)
                self.masks = torch.ones((2, 4, 5, 2))
                self.n_examples = 2
                self.n_train = 1
                self.n_test = 1
                self.shapes = [[[4, 5], [4, 5]], [[4, 5], [4, 5]]]
                self.in_out_same_size = True
                self.all_out_same_size = False
                self.all_in_same_size = False
        task = MockTask()
        train_step = 1
        initial_weight = model.weight.item()
        loss = take_step(task, model, optimizer, train_step, self.logger)
        optimizer.step()
        assert model.weight.item() != initial_weight

    def test_take_step_with_logging(self):
        """Test logging during training step"""
        class MockTask:
            def __init__(self):
                self.problem = torch.randint(0, 3, (2, 4, 5, 2), dtype=torch.long)
                self.solution = torch.randint(0, 3, (2, 4, 5), dtype=torch.long)
                self.masks = torch.ones((2, 4, 5, 2))
                self.n_examples = 2
                self.n_train = 1
                self.n_test = 1
                self.shapes = [[[4, 5], [4, 5]], [[4, 5], [4, 5]]]
                self.in_out_same_size = True
                self.all_out_same_size = False
                self.all_in_same_size = False
        task = MockTask()
        train_step = 1
        take_step(task, self.model, self.optimizer, train_step, self.logger)
        assert len(self.logger.logged_data) > 0

    def test_take_step_with_early_stopping(self):
        """Test early stopping during training"""
        class ConstantModel:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.randn(1))
                self.parameters = lambda: [self.weight]
                def forward():
                    logits = self.weight * torch.randn(2, 3, 4, 5, 2)
                    return (
                        logits,
                        torch.ones(2, 4, 2),
                        torch.ones(2, 5, 2),
                        [torch.tensor(0.1)],
                        ["kl_loss"]
                    )
                self.forward = forward
        model = ConstantModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        class MockTask:
            def __init__(self):
                self.problem = torch.randint(0, 3, (2, 4, 5, 2), dtype=torch.long)
                self.solution = torch.randint(0, 3, (2, 4, 5), dtype=torch.long)
                self.masks = torch.ones((2, 4, 5, 2))
                self.n_examples = 2
                self.n_train = 1
                self.n_test = 1
                self.shapes = [[[4, 5], [4, 5]], [[4, 5], [4, 5]]]
                self.in_out_same_size = True
                self.all_out_same_size = False
                self.all_in_same_size = False
        task = MockTask()
        train_step = 1
        take_step(task, model, optimizer, train_step, self.logger)
