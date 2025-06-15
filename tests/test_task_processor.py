import pytest
import numpy as np
import torch
import json
import tempfile
import os
import sys

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_task_processor import Task, preprocess_tasks

class TestTask:
    
    def test_task_initialization(self, sample_arc_problem, sample_arc_solution):
        """Test Task object initialization."""
        task = Task("test_task", sample_arc_problem, sample_arc_solution)
        
        assert task.task_name == "test_task"
        assert task.n_train == 2
        assert task.n_test == 1
        assert task.n_examples == 3
        assert task.unprocessed_problem == sample_arc_problem
        assert task.solution is not None
    
    def test_task_initialization_without_solution(self, sample_arc_problem):
        """Test Task initialization without solution."""
        task = Task("test_task", sample_arc_problem, None)
        
        assert task.task_name == "test_task"
        assert task.n_train == 2
        assert task.n_test == 1
        assert task.solution is None
    
    def test_collect_problem_shapes(self, sample_arc_problem):
        """Test shape collection from problem data."""
        task = Task("test_task", sample_arc_problem, None)
        
        expected_shapes = [
            [[2, 2], [2, 2]],  # train example 1
            [[2, 2], [2, 2]],  # train example 2
            [[2, 2], [2, 2]]   # test example (predicted output shape)
        ]
        
        assert task.shapes == expected_shapes
    
    def test_predict_solution_shapes_same_size(self, sample_arc_problem):
        """Test solution shape prediction when input/output are same size."""
        task = Task("test_task", sample_arc_problem, None)
        
        # Should predict test output shape same as input
        assert task.in_out_same_size == True
        assert task.shapes[2][1] == [2, 2]  # test output shape predicted
    
    def test_predict_solution_shapes_different_sizes(self):
        """Test solution shape prediction with different input/output sizes."""
        problem_diff_sizes = {
            'train': [
                {
                    'input': [[0, 1]],
                    'output': [[1, 0], [0, 1]]
                }
            ],
            'test': [
                {
                    'input': [[1, 2]]
                }
            ]
        }
        
        task = Task("test_task", problem_diff_sizes, None)
        
        assert task.in_out_same_size == False
        # Should predict max dimensions
        assert task.shapes[1][1] == [2, 2]  # predicted from max observed
    
    def test_construct_multitensor_system(self, sample_arc_problem):
        """Test multitensor system construction."""
        task = Task("test_task", sample_arc_problem, None)
        
        assert hasattr(task, 'multitensor_system')
        assert task.n_x >= 2  # at least as large as inputs
        assert task.n_y >= 2
        assert task.n_colors >= 3  # colors 0, 1, 2, 3 observed
        assert len(task.colors) >= 4  # including background
        assert 0 in task.colors  # background always included
    
    def test_create_grid_tensor(self, sample_arc_problem):
        """Test grid to tensor conversion."""
        task = Task("test_task", sample_arc_problem, None)
        
        grid = [[0, 1], [1, 0]]
        tensor = task._create_grid_tensor(grid)
        
        # Should create one-hot encoding for each color
        assert tensor.shape[0] == len(task.colors)
        assert tensor.shape[1:] == (2, 2)
        
        # Check one-hot encoding
        color_0_idx = task.colors.index(0)
        color_1_idx = task.colors.index(1)
        
        assert tensor[color_0_idx, 0, 0] == 1  # grid[0,0] = 0
        assert tensor[color_1_idx, 0, 0] == 0
        assert tensor[color_0_idx, 0, 1] == 0  # grid[0,1] = 1
        assert tensor[color_1_idx, 0, 1] == 1
    
    def test_create_problem_tensor(self, sample_arc_problem):
        """Test problem tensor creation."""
        task = Task("test_task", sample_arc_problem, None)
        
        assert hasattr(task, 'problem')
        assert isinstance(task.problem, torch.Tensor)
        
        # Shape should be (n_examples, n_x, n_y, 2)
        expected_shape = (task.n_examples, task.n_x, task.n_y, 2)
        assert task.problem.shape == expected_shape
    
    def test_compute_mask(self, sample_arc_problem):
        """Test mask computation."""
        task = Task("test_task", sample_arc_problem, None)
        
        assert hasattr(task, 'masks')
        assert isinstance(task.masks, torch.Tensor)
        
        # Shape should be (n_examples, n_x, n_y, 2)
        expected_shape = (task.n_examples, task.n_x, task.n_y, 2)
        assert task.masks.shape == expected_shape
        
        # Check that masks are binary
        assert torch.all((task.masks == 0) | (task.masks == 1))
    
    def test_create_solution_tensor(self, sample_arc_problem, sample_arc_solution):
        """Test solution tensor creation."""
        task = Task("test_task", sample_arc_problem, sample_arc_solution)
        
        assert hasattr(task, 'solution')
        assert isinstance(task.solution, torch.Tensor)
        assert hasattr(task, 'solution_hash')
        
        # Shape should be (n_test, n_x, n_y)
        expected_shape = (task.n_test, task.n_x, task.n_y)
        assert task.solution.shape == expected_shape


class TestTaskEdgeCases:
    
    def test_single_pixel_grids(self):
        """Test handling of single pixel grids."""
        problem = {
            'train': [{'input': [[5]], 'output': [[7]]}],
            'test': [{'input': [[5]]}]
        }
        
        task = Task("single_pixel", problem, None)
        assert task.n_x >= 1
        assert task.n_y >= 1
        assert 5 in task.colors
        assert 7 in task.colors
    
    def test_large_grids(self):
        """Test handling of larger grids."""
        large_grid = [[i % 10 for i in range(10)] for _ in range(10)]
        
        problem = {
            'train': [{'input': large_grid[:5], 'output': large_grid[:5]}],
            'test': [{'input': large_grid[:7]}]
        }
        
        task = Task("large_grid", problem, None)
        assert task.n_x >= 7
        assert task.n_y >= 10
    
    def test_all_colors(self):
        """Test handling of all 10 colors."""
        grid_all_colors = [[i for i in range(10)]]
        
        problem = {
            'train': [{'input': grid_all_colors, 'output': grid_all_colors}],
            'test': [{'input': grid_all_colors}]
        }
        
        task = Task("all_colors", problem, None)
        assert len(task.colors) == 10  # All colors 0-9
        assert set(task.colors) == set(range(10))
    
    def test_irregular_shapes(self):
        """Test handling of different shaped grids within same task."""
        problem = {
            'train': [
                {'input': [[0, 1]], 'output': [[1]]},  # 1x2 -> 1x1
                {'input': [[0], [1]], 'output': [[1], [0]]}  # 2x1 -> 2x1
            ],
            'test': [
                {'input': [[0, 1, 2]]}  # 1x3
            ]
        }
        
        task = Task("irregular", problem, None)
        assert task.n_x >= 2
        assert task.n_y >= 3
        assert not task.in_out_same_size
        assert not task.all_in_same_size
        assert not task.all_out_same_size


class TestPreprocessTasks:
    
    def test_preprocess_tasks_basic(self, temp_dir):
        """Test basic preprocessing functionality."""
        # Create dataset directory and files inline
        dataset_dir = os.path.join(temp_dir, 'dataset')
        os.makedirs(dataset_dir)
        problems = {
            'task1': {
                'train': [
                    {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
                    {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}
                ],
                'test': [
                    {'input': [[0, 1], [1, 0]]}
                ]
            },
            'task2': {
                'train': [
                    {'input': [[1, 2], [2, 1]], 'output': [[2, 1], [1, 2]]},
                    {'input': [[2, 1], [1, 2]], 'output': [[1, 2], [2, 1]]}
                ],
                'test': [
                    {'input': [[1, 2], [2, 1]]}
                ]
            }
        }
        solutions = {
            'task1': [[[1, 0], [0, 1]]],
            'task2': [[[2, 1], [1, 2]]]
        }
        with open(os.path.join(dataset_dir, 'arc-agi_training_challenges.json'), 'w') as f:
            json.dump(problems, f)
        with open(os.path.join(dataset_dir, 'arc-agi_training_solutions.json'), 'w') as f:
            json.dump(solutions, f)
        # Change working directory to dataset parent
        cwd = os.getcwd()
        os.chdir(os.path.dirname(dataset_dir))
        try:
            tasks = preprocess_tasks("training", ["task1", "task2"])
            assert len(tasks) == 2
            assert all(isinstance(task, Task) for task in tasks)
            assert tasks[0].task_name == "task1"
            assert tasks[1].task_name == "task2"
        finally:
            os.chdir(cwd)
    
    def test_preprocess_single_task(self, temp_dir):
        """Test preprocessing single task."""
        dataset_dir = os.path.join(temp_dir, 'dataset')
        os.makedirs(dataset_dir)
        problems = {
            'task1': {
                'train': [
                    {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
                    {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}
                ],
                'test': [
                    {'input': [[0, 1], [1, 0]]}
                ]
            }
        }
        solutions = {
            'task1': [[[1, 0], [0, 1]]]
        }
        with open(os.path.join(dataset_dir, 'arc-agi_training_challenges.json'), 'w') as f:
            json.dump(problems, f)
        with open(os.path.join(dataset_dir, 'arc-agi_training_solutions.json'), 'w') as f:
            json.dump(solutions, f)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(dataset_dir))
        try:
            tasks = preprocess_tasks("training", ["task1"])
            assert len(tasks) == 1
            assert tasks[0].task_name == "task1"
        finally:
            os.chdir(cwd)


class TestTaskIntegration:
    
    def test_full_task_pipeline(self, sample_arc_problem, sample_arc_solution):
        """Test complete task processing pipeline."""
        task = Task("integration_test", sample_arc_problem, sample_arc_solution)
        
        # Verify all components are properly initialized
        assert hasattr(task, 'multitensor_system')
        assert hasattr(task, 'problem')
        assert hasattr(task, 'masks')
        assert hasattr(task, 'solution')
        assert hasattr(task, 'shapes')
        assert hasattr(task, 'colors')
        
        # Verify tensor shapes are consistent
        assert task.problem.shape[0] == task.n_examples
        assert task.masks.shape[0] == task.n_examples
        assert task.solution.shape[0] == task.n_test
        
        # Verify all tensors are finite
        assert torch.isfinite(task.problem).all()
        assert torch.isfinite(task.masks).all()
        assert torch.isfinite(task.solution).all()
    
    def test_task_with_multitensor_operations(self, sample_arc_problem):
        """Test task works with multitensor system."""
        task = Task("multitensor_test", sample_arc_problem, None)
        
        # Create a simple multitensor
        mt = task.multitensor_system.make_multitensor(default=1.0)
        
        # Verify multitensor works with task
        assert mt is not None
        assert mt.multitensor_system == task.multitensor_system
        
        # Test basic operations
        for dims in task.multitensor_system:
            assert mt[dims] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])