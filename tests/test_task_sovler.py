import pytest
import numpy as np
import torch
import tempfile
import json
import time
import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_task_solver import solve_task_simple, batch_solve_tasks


class TestSolveTaskSimple:
    
    def setup_method(self):
        """Setup for each test method."""
        torch.set_default_device('cpu')  # Use CPU for testing
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Sample problem data
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
        
    def test_solve_task_simple_basic(self):
        """Test basic functionality of solve_task_simple."""
        solutions = solve_task_simple(
            task_name="test_task",
            problem_data=self.sample_problem,
            max_iterations=10,  # Small for testing
            time_limit=30
        )
        
        # Check return format
        assert isinstance(solutions, list)
        assert len(solutions) == 1  # One test example
        
        solution = solutions[0]
        assert isinstance(solution, dict)
        assert 'attempt_1' in solution
        assert 'attempt_2' in solution
        
        # Check attempts are lists of lists (grids)
        assert isinstance(solution['attempt_1'], list)
        assert isinstance(solution['attempt_2'], list)
        
        # Check attempts contain valid grid data
        for attempt in [solution['attempt_1'], solution['attempt_2']]:
            assert len(attempt) > 0  # Non-empty grid
            for row in attempt:
                assert isinstance(row, list)
                for cell in row:
                    assert isinstance(cell, (int, float))
                    assert 0 <= cell <= 9  # Valid ARC color
    
    def test_solve_task_simple_multiple_test_examples(self):
        """Test with multiple test examples."""
        problem_multi_test = {
            'train': [
                {
                    'input': [[0, 1]],
                    'output': [[1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[1, 2]]
                },
                {
                    'input': [[3, 4]]
                }
            ]
        }
        
        solutions = solve_task_simple(
            task_name="multi_test",
            problem_data=problem_multi_test,
            max_iterations=5,
            time_limit=10
        )
        
        # Should have solutions for both test examples
        assert len(solutions) == 2
        
        for i, solution in enumerate(solutions):
            assert 'attempt_1' in solution
            assert 'attempt_2' in solution
            
            # Each solution should be a valid grid
            for attempt in [solution['attempt_1'], solution['attempt_2']]:
                assert isinstance(attempt, list)
                assert len(attempt) > 0
    
    def test_solve_task_simple_time_limit(self):
        """Test that time limit is respected."""
        start_time = time.time()
        
        solutions = solve_task_simple(
            task_name="time_test",
            problem_data=self.sample_problem,
            max_iterations=1000,  # Large number
            time_limit=2  # Short time limit
        )
        
        elapsed = time.time() - start_time
        
        # Should respect time limit (with some tolerance)
        assert elapsed <= 5.0  # Allow some tolerance
        
        # Should still return valid solutions
        assert isinstance(solutions, list)
        assert len(solutions) > 0
    
    def test_solve_task_simple_early_stopping(self):
        """Test early stopping when solutions converge."""
        solutions = solve_task_simple(
            task_name="early_stop_test",
            problem_data=self.sample_problem,
            max_iterations=500,
            time_limit=60
        )
        
        # Should complete without timeout and return valid solutions
        assert isinstance(solutions, list)
        assert len(solutions) == 1
        assert 'attempt_1' in solutions[0]
        assert 'attempt_2' in solutions[0]
    
    def test_solve_task_simple_error_handling(self):
        """Test error handling with invalid input."""
        # Test with malformed problem data
        bad_problem = {
            'train': [],  # Empty training data
            'test': [{'input': [[0]]}]
        }
        
        # Should not crash, should return fallback solution
        solutions = solve_task_simple(
            task_name="bad_task",
            problem_data=bad_problem,
            max_iterations=5,
            time_limit=10
        )
        
        assert isinstance(solutions, list)
        assert len(solutions) > 0
        # Should have fallback solution
        assert solutions[0]['attempt_1'] == [[0]]
        assert solutions[0]['attempt_2'] == [[0]]
    
    def test_solve_task_simple_large_grid(self):
        """Test with larger grid size."""
        large_grid = [[i % 10 for i in range(10)] for _ in range(10)]
        
        large_problem = {
            'train': [
                {
                    'input': large_grid[:5],  # 5x10 input
                    'output': large_grid[:5]
                }
            ],
            'test': [
                {
                    'input': large_grid[:3]  # 3x10 test input
                }
            ]
        }
        
        solutions = solve_task_simple(
            task_name="large_grid",
            problem_data=large_problem,
            max_iterations=10,
            time_limit=30
        )
        
        assert isinstance(solutions, list)
        assert len(solutions) == 1
        
        # Check solution has reasonable dimensions
        solution = solutions[0]
        attempt_1 = solution['attempt_1']
        assert len(attempt_1) > 0
        assert len(attempt_1[0]) > 0


class TestBatchSolveTasks:
    
    def setup_method(self):
        """Setup for batch testing."""
        torch.set_default_device('cpu')
        
        # Create multiple sample problems
        self.task_list = [
            ("task1", {
                'train': [{'input': [[0, 1]], 'output': [[1, 0]]}],
                'test': [{'input': [[1, 1]]}]
            }),
            ("task2", {
                'train': [{'input': [[2, 3]], 'output': [[3, 2]]}],
                'test': [{'input': [[2, 2]]}]
            }),
            ("task3", {
                'train': [{'input': [[4, 5]], 'output': [[5, 4]]}],
                'test': [{'input': [[4, 4]]}]
            })
        ]
    
    def test_batch_solve_sequential(self):
        """Test batch solving with sequential execution."""
        solutions = batch_solve_tasks(
            task_list=self.task_list,
            split='test',
            max_workers=1,  # Sequential
            time_limit_per_task=10,
            max_iterations=5
        )
        
        # Check all tasks were solved
        assert isinstance(solutions, dict)
        assert len(solutions) == 3
        
        for task_name, _ in self.task_list:
            assert task_name in solutions
            task_solutions = solutions[task_name]
            assert isinstance(task_solutions, list)
            assert len(task_solutions) == 1  # One test example each
            
            solution = task_solutions[0]
            assert 'attempt_1' in solution
            assert 'attempt_2' in solution
    
    def test_batch_solve_empty_list(self):
        """Test batch solving with empty task list."""
        solutions = batch_solve_tasks(
            task_list=[],
            split='test',
            max_workers=1,
            time_limit_per_task=10,
            max_iterations=5
        )
        
        assert isinstance(solutions, dict)
        assert len(solutions) == 0
    
    def test_batch_solve_single_task(self):
        """Test batch solving with single task."""
        single_task = [self.task_list[0]]
        
        solutions = batch_solve_tasks(
            task_list=single_task,
            split='test',
            max_workers=1,
            time_limit_per_task=15,
            max_iterations=10
        )
        
        assert len(solutions) == 1
        assert "task1" in solutions
        assert len(solutions["task1"]) == 1


class TestSolveTaskEdgeCases:
    
    def test_minimal_task(self):
        """Test with minimal 1x1 task."""
        minimal_problem = {
            'train': [{'input': [[5]], 'output': [[7]]}],
            'test': [{'input': [[5]]}]
        }
        
        solutions = solve_task_simple(
            task_name="minimal",
            problem_data=minimal_problem,
            max_iterations=5,
            time_limit=10
        )
        
        assert len(solutions) == 1
        solution = solutions[0]
        
        # Check attempts are valid 
        for attempt in [solution['attempt_1'], solution['attempt_2']]:
            assert isinstance(attempt, list)
            assert len(attempt) >= 1
            assert len(attempt[0]) >= 1
            
            # Check values are valid colors
            for row in attempt:
                for cell in row:
                    assert 0 <= cell <= 9
    
    def test_no_training_data(self):
        """Test with no training examples."""
        no_train_problem = {
            'train': [],
            'test': [{'input': [[1, 2], [3, 4]]}]
        }
        
        # Should handle gracefully
        solutions = solve_task_simple(
            task_name="no_train",
            problem_data=no_train_problem,
            max_iterations=3,
            time_limit=5
        )
        
        assert isinstance(solutions, list)
        assert len(solutions) == 1
    
    def test_complex_patterns(self):
        """Test with more complex pattern task."""
        complex_problem = {
            'train': [
                {
                    'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    'output': [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
                },
                {
                    'input': [[0, 1], [2, 3]],
                    'output': [[3, 2], [1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[5, 6, 7], [8, 9, 0]]
                }
            ]
        }
        
        solutions = solve_task_simple(
            task_name="complex",
            problem_data=complex_problem,
            max_iterations=20,
            time_limit=30
        )
        
        assert len(solutions) == 1
        solution = solutions[0]
        
        # Check solutions have appropriate dimensions
        for attempt in [solution['attempt_1'], solution['attempt_2']]:
            assert len(attempt) >= 1
            assert len(attempt[0]) >= 1
    
    def test_all_same_color(self):
        """Test task where everything is the same color."""
        same_color_problem = {
            'train': [
                {
                    'input': [[0, 0], [0, 0]],
                    'output': [[0, 0], [0, 0]]
                }
            ],
            'test': [
                {
                    'input': [[0, 0, 0], [0, 0, 0]]
                }
            ]
        }
        
        solutions = solve_task_simple(
            task_name="same_color",
            problem_data=same_color_problem,
            max_iterations=5,
            time_limit=10
        )
        
        assert len(solutions) == 1
        solution = solutions[0]
        
        # Should produce valid solutions
        for attempt in [solution['attempt_1'], solution['attempt_2']]:
            assert isinstance(attempt, list)
            for row in attempt:
                for cell in row:
                    assert isinstance(cell, (int, float))
                    assert 0 <= cell <= 9


class TestSolveTaskPerformance:
    
    def test_reasonable_execution_time(self):
        """Test that solving doesn't take excessively long."""
        problem = {
            'train': [
                {
                    'input': [[0, 1, 2], [3, 4, 5]],
                    'output': [[5, 4, 3], [2, 1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[6, 7, 8], [9, 0, 1]]
                }
            ]
        }
        
        start_time = time.time()
        
        solutions = solve_task_simple(
            task_name="performance_test",
            problem_data=problem,
            max_iterations=50,
            time_limit=20
        )
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed <= 25.0  # Allow some buffer
        assert isinstance(solutions, list)
        assert len(solutions) == 1
    
    def test_memory_efficiency(self):
        """Test that solving doesn't consume excessive memory."""
        # Create medium-sized problem
        medium_grid = [[i % 10 for i in range(8)] for _ in range(8)]
        
        medium_problem = {
            'train': [
                {
                    'input': medium_grid,
                    'output': medium_grid
                }
            ] * 2,  # 2 training examples
            'test': [
                {
                    'input': medium_grid
                }
            ]
        }
        
        # Should complete without memory issues
        solutions = solve_task_simple(
            task_name="memory_test",
            problem_data=medium_problem,
            max_iterations=20,
            time_limit=30
        )
        
        assert isinstance(solutions, list)
        assert len(solutions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])