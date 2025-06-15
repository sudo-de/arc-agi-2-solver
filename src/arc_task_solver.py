import os
import sys
import time
import json
import gc
import multiprocessing
import traceback
import numpy as np
import torch

# Import all required modules
from src.arc_task_processor import Task, preprocess_tasks
from src.arc_compressor import ARCCompressor
from src.arc_weight_initializer import Initializer
from src.arc_multitensor import multitensor_systems
from src.arc_network_layers import layers
from src.arc_solution_selector import Logger
from src.arc_visualizer import plot_problem, plot_solution
from src.arc_trainer import take_step


def solve_task(task_name, split, time_limit, n_train_iterations, gpu_id, memory_dict, solutions_dict, error_queue):
    """
    Solve a single ARC task using the VAE decoder approach.
    
    Args:
        task_name: Name of the task to solve
        split: Dataset split ('test', 'evaluation', 'training')
        time_limit: Maximum time allowed for solving
        n_train_iterations: Number of training iterations
        gpu_id: GPU device ID to use
        memory_dict: Shared dictionary for memory usage tracking
        solutions_dict: Shared dictionary for storing solutions
        error_queue: Queue for error reporting
    """
    try:
        # Set device and reset memory stats
        torch.set_default_device('cuda')
        torch.cuda.set_device(gpu_id)
        torch.cuda.reset_peak_memory_stats()
        
        # Load the task data
        with open(f'/kaggle/input/arc-prize-2025/arc-agi_{split}_challenges.json', 'r') as f:
            problems = json.load(f)
        
        # Create task object
        task = Task(task_name, problems[task_name], None)
        del problems
        
        # Initialize model and optimizer
        model = ARCCompressor(task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        train_history_logger = Logger(task)
        
        # Initialize default solutions
        default_solution = tuple(((0, 0), (0, 0)) for _ in range(task.n_test))
        train_history_logger.solution_most_frequent = default_solution
        train_history_logger.solution_second_most_frequent = default_solution
        
        # Training loop
        for train_step in range(n_train_iterations):
            take_step(task, model, optimizer, train_step, train_history_logger)
            
            # Check time limit
            if time.time() > time_limit:
                print(f"Time limit reached for task {task_name} at step {train_step}")
                break
            
            # Early stopping if solutions converge
            if train_step > 100 and train_step % 50 == 0:
                if (train_history_logger.solution_most_frequent is not None and 
                    train_history_logger.solution_second_most_frequent is not None):
                    # Check if solutions are reasonable (not just default)
                    if (len(train_history_logger.solution_most_frequent) > 0 and
                        train_history_logger.solution_most_frequent != default_solution):
                        print(f"Converged solution found for {task_name} at step {train_step}")
                        break
        
        # Extract solutions
        example_list = []
        for example_num in range(task.n_test):
            if (train_history_logger.solution_most_frequent is not None and 
                len(train_history_logger.solution_most_frequent) > example_num):
                attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            else:
                attempt_1 = [[0]]
                
            if (train_history_logger.solution_second_most_frequent is not None and 
                len(train_history_logger.solution_second_most_frequent) > example_num):
                attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            else:
                attempt_2 = [[0]]
                
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})
        
        # Clean up memory
        del task, model, optimizer, train_history_logger
        torch.cuda.empty_cache()
        gc.collect()
        
        # Store results
        memory_dict[task_name] = torch.cuda.max_memory_allocated()
        solutions_dict[task_name] = example_list
        
        print(f"Successfully solved task {task_name}")
        
    except Exception as e:
        error_message = f"Error in task {task_name}: {traceback.format_exc()}"
        print(error_message)
        error_queue.put(error_message)
        
        # Ensure we always provide a fallback solution
        if task_name not in solutions_dict:
            solutions_dict[task_name] = [{'attempt_1': [[0]], 'attempt_2': [[0]]}]


def solve_task_simple(task_name, problem_data, max_iterations=500, time_limit=60):
    """
    Simplified version for single-process execution (Kaggle environment).
    
    Args:
        task_name: Name of the task
        problem_data: The task data dict with 'train' and 'test' keys
        max_iterations: Maximum training iterations
        time_limit: Time limit in seconds
    
    Returns:
        List of solution dictionaries with 'attempt_1' and 'attempt_2'
    """
    try:
        start_time = time.time()
        
        # Create task object
        task = Task(task_name, problem_data, None)
        
        # Initialize model
        model = ARCCompressor(task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        logger = Logger(task)
        
        # Initialize default solutions (fallback)
        default_solution = tuple(((0, 0), (0, 0)) for _ in range(task.n_test))
        logger.solution_most_frequent = default_solution
        logger.solution_second_most_frequent = default_solution
        
        # Training loop
        for step in range(max_iterations):
            if time.time() - start_time > time_limit:
                print(f"Time limit reached for {task_name} at step {step}")
                break
                
            take_step(task, model, optimizer, step, logger)
            
            # Early stopping if solutions converge
            if step > 100 and step % 50 == 0:
                if (logger.solution_most_frequent is not None and 
                    logger.solution_second_most_frequent is not None and
                    logger.solution_most_frequent != default_solution):
                    print(f"Solutions converged for {task_name} at step {step}")
                    break
        
        # Extract solutions
        solutions = []
        for example_num in range(task.n_test):
            if (logger.solution_most_frequent is not None and 
                len(logger.solution_most_frequent) > example_num):
                attempt_1 = [list(row) for row in logger.solution_most_frequent[example_num]]
            else:
                attempt_1 = [[0]]
                
            if (logger.solution_second_most_frequent is not None and 
                len(logger.solution_second_most_frequent) > example_num):
                attempt_2 = [list(row) for row in logger.solution_second_most_frequent[example_num]]
            else:
                attempt_2 = [[0]]
                
            solutions.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})
        
        # Cleanup
        del task, model, optimizer, logger
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"Task {task_name} completed in {elapsed:.1f}s")
        
        return solutions
        
    except Exception as e:
        print(f"Error solving {task_name}: {e}")
        traceback.print_exc()
        # Return minimal fallback solution
        return [{'attempt_1': [[0]], 'attempt_2': [[0]]}]


def batch_solve_tasks(task_list, split='test', max_workers=1, time_limit_per_task=60, max_iterations=500):
    """
    Solve multiple tasks either sequentially or in parallel.
    
    Args:
        task_list: List of (task_name, problem_data) tuples
        split: Dataset split
        max_workers: Number of parallel workers (1 for sequential)
        time_limit_per_task: Time limit per task in seconds
        max_iterations: Maximum training iterations per task
    
    Returns:
        Dictionary mapping task names to solutions
    """
    all_solutions = {}
    
    if max_workers == 1:
        # Sequential execution (recommended for Kaggle)
        for i, (task_name, problem_data) in enumerate(task_list):
            print(f"Solving task {i+1}/{len(task_list)}: {task_name}")
            solutions = solve_task_simple(
                task_name, 
                problem_data, 
                max_iterations=max_iterations,
                time_limit=time_limit_per_task
            )
            all_solutions[task_name] = solutions
    else:
        # Parallel execution (may cause issues in Kaggle)
        print("Warning: Parallel execution may cause issues in Kaggle environment")
        manager = multiprocessing.Manager()
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()
        
        processes = []
        for task_name, problem_data in task_list:
            # Save problem data temporarily for multiprocessing
            temp_file = f"/tmp/{task_name}.json"
            with open(temp_file, 'w') as f:
                json.dump(problem_data, f)
            
            p = multiprocessing.Process(
                target=solve_task,
                args=(task_name, split, time.time() + time_limit_per_task, 
                      max_iterations, 0, memory_dict, solutions_dict, error_queue)
            )
            processes.append(p)
            p.start()
            
            # Limit concurrent processes
            if len(processes) >= max_workers:
                for proc in processes:
                    proc.join()
                processes = []
        
        # Wait for remaining processes
        for proc in processes:
            proc.join()
        
        # Collect results
        all_solutions = dict(solutions_dict)
        
        # Report errors
        while not error_queue.empty():
            print(error_queue.get())
    
    return all_solutions


if __name__ == "__main__":
    # Example usage
    print("solve_task.py - ARC-AGI Task Solver")
    print("This module provides functions to solve ARC tasks using the VAE decoder approach.")
    print("Use solve_task_simple() for single tasks or batch_solve_tasks() for multiple tasks.")