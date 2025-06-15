import pytest
import numpy as np
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_visualizer import convert_color, plot_problem, plot_solution
from src.arc_task_processor import Task

class TestVisualizer:
    
    def setup_method(self):
        """Setup for each test method."""
        # Create sample problem for testing
        self.sample_problem = {
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
        
        # Create task
        self.task = Task("test_task", self.sample_problem, None)
        
        # Create mock logger with all attributes needed for plot_solution
        self.mock_logger = MagicMock()
        self.mock_logger.task = self.task
        self.mock_logger.logits = torch.randn(2, 10, 2, 2, 2)  # [example, color, x, y, in/out]
        self.mock_logger.x_mask = torch.randn(2, 2, 2)  # [example, x, in/out]
        self.mock_logger.y_mask = torch.randn(2, 2, 2)  # [example, y, in/out]
        
        # For plot_solution
        n_test = self.task.n_test
        n_colors = self.task.n_colors
        n_x = self.task.n_x
        n_y = self.task.n_y
        
        # Create tensors with proper shapes
        shape = (n_test, n_colors + 1, n_x, n_y)
        self.mock_logger.current_logits = torch.randn(shape)
        self.mock_logger.ema_logits = torch.randn(shape)
        self.mock_logger.current_x_mask = torch.randn(n_test, n_x)
        self.mock_logger.current_y_mask = torch.randn(n_test, n_y)
        self.mock_logger.ema_x_mask = torch.randn(n_test, n_x)
        self.mock_logger.ema_y_mask = torch.randn(n_test, n_y)
        
        # Create solution tensors with proper shapes
        self.mock_logger.solution_most_frequent = tuple(
            tuple(tuple(0 for _ in range(n_y)) for _ in range(n_x)) 
            for _ in range(n_test)
        )
        self.mock_logger.solution_second_most_frequent = tuple(
            tuple(tuple(0 for _ in range(n_y)) for _ in range(n_x)) 
            for _ in range(n_test)
        )
        
        # Add any missing attributes that might be needed
        self.mock_logger.task.n_test = n_test
        self.mock_logger.task.n_colors = n_colors
        self.mock_logger.task.n_x = n_x
        self.mock_logger.task.n_y = n_y
        self.mock_logger.task.shapes = [
            (self.task.shapes[0][0], self.task.shapes[0][1]),
            (self.task.shapes[1][0], self.task.shapes[1][1])
        ]

    def test_convert_color(self):
        """Test color conversion function with one-hot input."""
        # Test with single color (one-hot)
        grid = np.zeros((1, 1, 10), dtype=np.float32)
        grid[0, 0, 0] = 1
        result = convert_color(grid)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1, 3)  # RGB format
        
        # Test with multiple colors (one-hot)
        grid = np.zeros((2, 2, 10), dtype=np.float32)
        grid[0, 0, 0] = 1
        grid[0, 1, 1] = 1
        grid[1, 0, 2] = 1
        grid[1, 1, 3] = 1
        result = convert_color(grid)
        assert result.shape == (2, 2, 3)
        
        # Test with invalid color (index out of range in one-hot)
        grid = np.zeros((1, 1, 11), dtype=np.float32)  # 11 channels, should fail
        with pytest.raises(ValueError):
            convert_color(grid)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_problem(self, mock_show):
        """Test problem plotting."""
        # Test plotting without errors (do not assert show is called)
        plot_problem(self.mock_logger)
        # No assertion for mock_show, as plot_problem only calls savefig
        
        # Test with invalid logger
        with pytest.raises(AttributeError):
            plot_problem(None)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_solution(self, mock_savefig):
        """Test solution plotting."""
        # Test plotting without errors
        plot_solution(self.mock_logger)
        # Just check savefig was called at least once
        assert mock_savefig.call_count >= 1
        
        # Test with custom filename
        plot_solution(self.mock_logger, fname='test.png')
        assert mock_savefig.call_count >= 2
        
        # Test with invalid logger
        with pytest.raises(AttributeError):
            plot_solution(None)
    
    def test_plot_edge_cases(self):
        """Test plotting with edge cases (use at least 1x1 grid)."""
        # Test with minimal 1x1 grid
        minimal_problem = {
            'train': [{'input': [[0]], 'output': [[0]]}],
            'test': [{'input': [[0]]}]
        }
        minimal_task = Task("minimal_task", minimal_problem, None)
        minimal_logger = MagicMock()
        minimal_logger.task = minimal_task
        minimal_logger.logits = torch.randn(1, 10, 1, 1, 2)
        minimal_logger.x_mask = torch.randn(1, 1, 2)
        minimal_logger.y_mask = torch.randn(1, 1, 2)
        
        # For plot_solution
        n_test = minimal_task.n_test
        n_colors = minimal_task.n_colors
        n_x = minimal_task.n_x
        n_y = minimal_task.n_y
        shape = (n_test, n_colors + 1, n_x, n_y)
        minimal_logger.current_logits = torch.randn(shape)
        minimal_logger.ema_logits = torch.randn(shape)
        minimal_logger.current_x_mask = torch.randn(n_test, n_x)
        minimal_logger.current_y_mask = torch.randn(n_test, n_y)
        minimal_logger.ema_x_mask = torch.randn(n_test, n_x)
        minimal_logger.ema_y_mask = torch.randn(n_test, n_y)
        minimal_logger.solution_most_frequent = tuple(
            tuple(tuple(0 for _ in range(n_y)) for _ in range(n_x)) 
            for _ in range(n_test)
        )
        minimal_logger.solution_second_most_frequent = tuple(
            tuple(tuple(0 for _ in range(n_y)) for _ in range(n_x)) 
            for _ in range(n_test)
        )
        
        # Add any missing attributes that might be needed
        minimal_logger.task.n_test = n_test
        minimal_logger.task.n_colors = n_colors
        minimal_logger.task.n_x = n_x
        minimal_logger.task.n_y = n_y
        minimal_logger.task.shapes = [
            (minimal_task.shapes[0][0], minimal_task.shapes[0][1]),
            (minimal_task.shapes[1][0], minimal_task.shapes[1][1])
        ]
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            plot_problem(minimal_logger)
            plot_solution(minimal_logger)
        
        # Test with large grid
        large_problem = {
            'train': [{'input': [[0]*10]*10, 'output': [[0]*10]*10}],
            'test': [{'input': [[0]*10]*10}]
        }
        large_task = Task("large_task", large_problem, None)
        large_logger = MagicMock()
        large_logger.task = large_task
        large_logger.logits = torch.randn(1, 10, 10, 10, 2)
        large_logger.x_mask = torch.randn(1, 10, 2)
        large_logger.y_mask = torch.randn(1, 10, 2)
        
        n_test = large_task.n_test
        n_colors = large_task.n_colors
        n_x = large_task.n_x
        n_y = large_task.n_y
        shape = (n_test, n_colors + 1, n_x, n_y)
        large_logger.current_logits = torch.randn(shape)
        large_logger.ema_logits = torch.randn(shape)
        large_logger.current_x_mask = torch.randn(n_test, n_x)
        large_logger.current_y_mask = torch.randn(n_test, n_y)
        large_logger.ema_x_mask = torch.randn(n_test, n_x)
        large_logger.ema_y_mask = torch.randn(n_test, n_y)
        large_logger.solution_most_frequent = tuple(
            tuple(tuple(0 for _ in range(n_y)) for _ in range(n_x)) 
            for _ in range(n_test)
        )
        large_logger.solution_second_most_frequent = tuple(
            tuple(tuple(0 for _ in range(n_y)) for _ in range(n_x)) 
            for _ in range(n_test)
        )
        
        # Add any missing attributes that might be needed
        large_logger.task.n_test = n_test
        large_logger.task.n_colors = n_colors
        large_logger.task.n_x = n_x
        large_logger.task.n_y = n_y
        large_logger.task.shapes = [
            (large_task.shapes[0][0], large_task.shapes[0][1]),
            (large_task.shapes[1][0], large_task.shapes[1][1])
        ]
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            plot_problem(large_logger)
            plot_solution(large_logger)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
