import os
import sys
import time
import json
import gc
import traceback
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
import torch
import warnings
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SolvingConfig:
    """Advanced configuration for task solving."""
    time_limit_per_task: float = 300.0
    n_train_iterations: int = 2000
    max_workers: int = 4
    use_ensemble: bool = True
    ensemble_size: int = 5
    adaptive_iterations: bool = True
    save_visualizations: bool = False
    temperature_scaling: bool = True
    uncertainty_threshold: float = 0.15
    quality_threshold: float = 0.3
    early_stopping_patience: int = 100
    memory_limit_gb: float = 8.0
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_iterations: int = 50
    use_adaptive_lr: bool = True
    solution_diversity_weight: float = 0.3
    confidence_boost_threshold: float = 0.8


@dataclass
class TaskResult:
    """Comprehensive result from solving a single task."""
    task_name: str
    solutions: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool
    solving_time: float
    peak_memory_mb: float
    iterations_completed: int
    convergence_step: Optional[int] = None
    confidence_scores: Optional[List[float]] = None
    error_message: Optional[str] = None


class MemoryManager:
    """Advanced memory management for efficient resource usage."""
    
    def __init__(self, limit_gb: float = 8.0):
        self.limit_bytes = limit_gb * 1024**3
        self.peak_usage = 0
        
    def check_memory(self) -> Tuple[bool, float]:
        """Check current memory usage."""
        current_usage = psutil.Process().memory_info().rss
        self.peak_usage = max(self.peak_usage, current_usage)
        
        return current_usage < self.limit_bytes, current_usage / 1024**2
    
    def cleanup(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class PerformanceTracker:
    """Track and analyze solving performance across tasks."""
    
    def __init__(self):
        self.task_times = {}
        self.task_accuracies = {}
        self.task_complexities = {}
        self.solution_patterns = defaultdict(int)
        
    def update(self, task_name: str, result: TaskResult):
        """Update performance metrics."""
        self.task_times[task_name] = result.solving_time
        self.task_complexities[task_name] = result.metadata.get('complexity', 0.5)
        
        # Track solution patterns
        for sol in result.solutions:
            pattern_key = self._extract_pattern(sol)
            self.solution_patterns[pattern_key] += 1
    
    def _extract_pattern(self, solution: Dict) -> str:
        """Extract pattern signature from solution."""
        try:
            attempt = solution.get('attempt_1', [[0]])
            h, w = len(attempt), len(attempt[0]) if attempt else 1
            colors = set()
            for row in attempt:
                colors.update(row)
            return f"size_{h}x{w}_colors_{len(colors)}"
        except:
            return "unknown"
    
    def get_adaptive_config(self, task_complexity: float) -> Dict[str, Any]:
        """Get adaptive configuration based on performance history."""
        base_iterations = 2000
        
        # Adjust based on complexity
        if task_complexity > 0.7:
            iterations = int(base_iterations * 1.5)
            lr_scale = 0.8
        elif task_complexity < 0.3:
            iterations = int(base_iterations * 0.7)
            lr_scale = 1.2
        else:
            iterations = base_iterations
            lr_scale = 1.0
        
        return {
            'n_iterations': iterations,
            'lr_scale': lr_scale,
            'ensemble_size': min(5, max(3, int(task_complexity * 10)))
        }


class EnsembleSolver:
    """Advanced ensemble solver with multiple strategies."""
    
    def __init__(self, config: SolvingConfig):
        self.config = config
        self.memory_manager = MemoryManager(config.memory_limit_gb)
        
    def solve_with_ensemble(self, task_name: str, task_data: Dict[str, Any]) -> TaskResult:
        """Solve task using ensemble of different approaches."""
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from . import preprocessing
            from . import arc_compressor
            from . import train
            from . import solution_selection
            
            # Create task
            task = preprocessing.EnhancedTask(
                task_name, task_data, None, 
                use_augmentation=True, 
                analyze_complexity=True
            )
            
            # Get adaptive configuration
            adaptive_config = task.get_adaptive_config()
            ensemble_size = adaptive_config.get('ensemble_size', self.config.ensemble_size)
            
            # Solve with multiple models
            ensemble_results = []
            
            for ensemble_idx in range(ensemble_size):
                try:
                    result = self._solve_single_model(
                        task, ensemble_idx, adaptive_config
                    )
                    ensemble_results.append(result)
                    
                    # Memory check
                    memory_ok, memory_mb = self.memory_manager.check_memory()
                    if not memory_ok:
                        logger.warning(f"Memory limit exceeded: {memory_mb:.1f}MB")
                        break
                        
                except Exception as e:
                    logger.warning(f"Ensemble {ensemble_idx} failed: {e}")
                    continue
            
            # Combine ensemble results
            final_solutions = self._combine_ensemble_results(
                ensemble_results, task.n_test
            )
            
            # Calculate metadata
            solving_time = time.time() - start_time
            metadata = {
                'complexity': task.stats.difficulty_score if task.stats else 0.5,
                'ensemble_size': len(ensemble_results),
                'adaptive_config': adaptive_config,
                'convergence_analysis': self._analyze_convergence(ensemble_results)
            }
            
            return TaskResult(
                task_name=task_name,
                solutions=final_solutions,
                metadata=metadata,
                success=True,
                solving_time=solving_time,
                peak_memory_mb=self.memory_manager.peak_usage / 1024**2,
                iterations_completed=sum(r.get('iterations', 0) for r in ensemble_results),
                confidence_scores=self._calculate_confidence_scores(ensemble_results)
            )
            
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Task {task_name} failed: {error_msg}")
            
            return TaskResult(
                task_name=task_name,
                solutions=self._create_fallback_solutions(task_data),
                metadata={'error': str(e)},
                success=False,
                solving_time=time.time() - start_time,
                peak_memory_mb=self.memory_manager.peak_usage / 1024**2,
                iterations_completed=0,
                error_message=error_msg
            )
        
        finally:
            self.memory_manager.cleanup()
    
    def _solve_single_model(self, task, ensemble_idx: int, adaptive_config: Dict) -> Dict[str, Any]:
        """Solve with a single model configuration."""
        
        # Import here to avoid circular imports
        from . import arc_compressor
        from . import train
        from . import solution_selection
        
        # Create model with ensemble-specific configuration
        model_config = {
            'variant': 'competition' if ensemble_idx == 0 else 'advanced',
            'base_dim': 64 if ensemble_idx == 0 else 48,
            'n_layers': adaptive_config.get('n_layers', 6),
            'ensemble_index': ensemble_idx,
            'use_attention': adaptive_config.get('use_attention', True),
            'dropout': 0.1 + 0.05 * ensemble_idx  # Slight variation
        }
        
        model = arc_compressor.create_arc_compressor(task, **model_config)
        
        # Enhanced optimizer
        optimizer = self._create_enhanced_optimizer(model, adaptive_config)
        
        # Enhanced logger
        logger_config = {
            'uncertainty_estimation': True,
            'solution_tracking': True,
            'ensemble_id': ensemble_idx
        }
        train_logger = solution_selection.EnhancedLogger(task, logger_config)
        
        # Training with advanced techniques
        n_iterations = adaptive_config.get('n_iterations', self.config.n_train_iterations)
        
        convergence_step = None
        best_loss = float('inf')
        patience_counter = 0
        
        for train_step in range(n_iterations):
            try:
                # Enhanced training step
                metrics = train.enhanced_take_step(
                    task, model, optimizer, train_step, train_logger
                )
                
                # Early stopping check
                current_loss = metrics['total_loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (patience_counter >= self.config.early_stopping_patience and 
                    train_step > self.config.warmup_iterations):
                    convergence_step = train_step
                    break
                
                # Memory check every 100 steps
                if train_step % 100 == 0:
                    memory_ok, _ = self.memory_manager.check_memory()
                    if not memory_ok:
                        break
                        
            except Exception as e:
                logger.warning(f"Training step {train_step} failed: {e}")
                break
        
        # Extract best solutions
        best_solutions = train_logger.get_ensemble_solutions(n_solutions=4)
        
        return {
            'solutions': best_solutions,
            'final_loss': best_loss,
            'convergence_step': convergence_step,
            'iterations': train_step + 1,
            'ensemble_idx': ensemble_idx,
            'quality_scores': [
                q.overall_score for q in train_logger.solution_tracker.quality_scores[-10:]
            ] if hasattr(train_logger, 'solution_tracker') else []
        }
    
    def _create_enhanced_optimizer(self, model, adaptive_config: Dict):
        """Create enhanced optimizer with adaptive settings."""
        
        # Import here to avoid circular imports
        from . import train
        
        base_lr = 0.01 * adaptive_config.get('lr_scale', 1.0)
        
        config = {
            'base_lr': base_lr,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'use_adaptive_lr': True,
            'warmup_steps': self.config.warmup_iterations
        }
        
        optimizer, scheduler = train.create_optimizer_and_scheduler(model, config)
        return optimizer
    
    def _combine_ensemble_results(self, ensemble_results: List[Dict], n_test: int) -> List[Dict[str, Any]]:
        """Combine results from ensemble of models using advanced voting."""
        
        if not ensemble_results:
            return [{'attempt_1': [[0, 0]], 'attempt_2': [[0, 0]]} for _ in range(n_test)]
        
        final_solutions = []
        
        for test_idx in range(n_test):
            # Collect all candidate solutions for this test case
            candidates = []
            weights = []
            
            for result in ensemble_results:
                if 'solutions' in result and test_idx < len(result['solutions']):
                    solution = result['solutions'][test_idx]
                    
                    # Weight based on model quality
                    weight = 1.0
                    if 'quality_scores' in result and result['quality_scores']:
                        weight = np.mean(result['quality_scores'])
                    
                    # Bonus for convergence
                    if result.get('convergence_step') is not None:
                        weight *= 1.2
                    
                    candidates.append(solution)
                    weights.append(weight)
            
            if not candidates:
                final_solutions.append({'attempt_1': [[0, 0]], 'attempt_2': [[0, 0]]})
                continue
            
            # Advanced ensemble combination
            combined_solution = self._advanced_solution_combination(
                candidates, weights, test_idx
            )
            final_solutions.append(combined_solution)
        
        return final_solutions
    
    def _advanced_solution_combination(self, candidates: List, weights: List[float], 
                                     test_idx: int) -> Dict[str, Any]:
        """Advanced solution combination with confidence weighting."""
        
        if len(candidates) == 1:
            solution = candidates[0]
            return {
                'attempt_1': self._ensure_valid_grid(solution),
                'attempt_2': self._ensure_valid_grid(solution)
            }
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(candidates)] * len(candidates)
        
        # Find most common solutions
        solution_counts = Counter()
        solution_weights = defaultdict(float)
        
        for candidate, weight in zip(candidates, weights):
            try:
                # Convert to hashable format
                grid = self._ensure_valid_grid(candidate)
                grid_tuple = tuple(tuple(row) for row in grid)
                
                solution_counts[grid_tuple] += 1
                solution_weights[grid_tuple] += weight
                
            except Exception:
                continue
        
        if not solution_counts:
            return {'attempt_1': [[0, 0]], 'attempt_2': [[0, 0]]}
        
        # Sort by combined score (frequency + weight)
        scored_solutions = []
        for grid_tuple, count in solution_counts.items():
            weight_score = solution_weights[grid_tuple]
            diversity_score = self._calculate_diversity_score(grid_tuple)
            
            combined_score = (0.5 * count + 
                            0.4 * weight_score + 
                            0.1 * diversity_score)
            
            scored_solutions.append((combined_score, grid_tuple))
        
        scored_solutions.sort(reverse=True)
        
        # Select top 2 solutions
        attempt_1 = [list(row) for row in scored_solutions[0][1]]
        
        if len(scored_solutions) > 1:
            attempt_2 = [list(row) for row in scored_solutions[1][1]]
        else:
            # Create slight variation if only one solution
            attempt_2 = self._create_solution_variation(attempt_1)
        
        return {
            'attempt_1': attempt_1,
            'attempt_2': attempt_2
        }
    
    def _ensure_valid_grid(self, candidate) -> List[List[int]]:
        """Ensure solution is a valid grid format."""
        
        try:
            if isinstance(candidate, dict):
                # Extract from attempt_1 or first available key
                if 'attempt_1' in candidate:
                    grid = candidate['attempt_1']
                elif 'attempt_2' in candidate:
                    grid = candidate['attempt_2']
                else:
                    # Get first available value
                    grid = next(iter(candidate.values()))
            else:
                grid = candidate
            
            # Ensure it's a list of lists
            if not isinstance(grid, list):
                return [[0, 0]]
            
            if not grid or not isinstance(grid[0], list):
                return [[0, 0]]
            
            # Validate and clamp values
            valid_grid = []
            for row in grid:
                if isinstance(row, list):
                    valid_row = [max(0, min(9, int(cell))) if isinstance(cell, (int, float)) else 0 
                                for cell in row]
                    valid_grid.append(valid_row)
                else:
                    valid_grid.append([0])
            
            # Ensure non-empty
            if not valid_grid:
                return [[0, 0]]
            
            return valid_grid
            
        except Exception:
            return [[0, 0]]
    
    def _calculate_diversity_score(self, grid_tuple: Tuple) -> float:
        """Calculate diversity score for solution selection."""
        try:
            grid = np.array([list(row) for row in grid_tuple])
            
            # Measures of diversity
            unique_values = len(np.unique(grid))
            entropy = self._calculate_entropy(grid)
            spatial_complexity = self._calculate_spatial_complexity(grid)
            
            # Combine metrics
            diversity = (0.4 * unique_values / 10.0 + 
                        0.3 * entropy + 
                        0.3 * spatial_complexity)
            
            return min(1.0, diversity)
            
        except Exception:
            return 0.0
    
    def _calculate_entropy(self, grid: np.ndarray) -> float:
        """Calculate normalized entropy of grid."""
        try:
            values, counts = np.unique(grid, return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(values)) if len(values) > 1 else 1
            return entropy / max_entropy
        except:
            return 0.0
    
    def _calculate_spatial_complexity(self, grid: np.ndarray) -> float:
        """Calculate spatial complexity of grid."""
        try:
            if grid.size < 4:
                return 0.0
            
            # Count different local patterns
            patterns = set()
            for i in range(grid.shape[0] - 1):
                for j in range(grid.shape[1] - 1):
                    pattern = tuple(grid[i:i+2, j:j+2].flatten())
                    patterns.add(pattern)
            
            max_patterns = min(len(patterns), 4**4)
            return len(patterns) / max(max_patterns, 1)
            
        except:
            return 0.0
    
    def _create_solution_variation(self, base_solution: List[List[int]]) -> List[List[int]]:
        """Create a slight variation of the base solution."""
        try:
            variation = [row[:] for row in base_solution]  # Deep copy
            
            # Apply small random changes
            if len(variation) > 1 and len(variation[0]) > 1:
                # Randomly modify 1 cell
                row_idx = np.random.randint(0, len(variation))
                col_idx = np.random.randint(0, len(variation[0]))
                
                # Cycle through colors
                current_val = variation[row_idx][col_idx]
                variation[row_idx][col_idx] = (current_val + 1) % 10
            
            return variation
            
        except Exception:
            return base_solution
    
    def _create_fallback_solutions(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback solutions when solving fails."""
        
        n_test = len(task_data.get('test', []))
        fallback_solutions = []
        
        for test_idx in range(n_test):
            # Try to infer output size from training examples
            try:
                train_examples = task_data.get('train', [])
                if train_examples:
                    # Use most common output size
                    output_shapes = [
                        (len(ex['output']), len(ex['output'][0])) 
                        for ex in train_examples if 'output' in ex
                    ]
                    if output_shapes:
                        most_common_shape = Counter(output_shapes).most_common(1)[0][0]
                        h, w = most_common_shape
                        
                        # Create simple pattern
                        grid = [[0 for _ in range(w)] for _ in range(h)]
                        fallback_solutions.append({
                            'attempt_1': grid,
                            'attempt_2': grid
                        })
                        continue
            except Exception:
                pass
            
            # Ultimate fallback
            fallback_solutions.append({
                'attempt_1': [[0, 0]],
                'attempt_2': [[0, 0]]
            })
        
        return fallback_solutions
    
    def _analyze_convergence(self, ensemble_results: List[Dict]) -> Dict[str, Any]:
        """Analyze convergence patterns across ensemble."""
        
        convergence_steps = [
            r.get('convergence_step') for r in ensemble_results 
            if r.get('convergence_step') is not None
        ]
        
        final_losses = [
            r.get('final_loss') for r in ensemble_results 
            if r.get('final_loss') is not None
        ]
        
        return {
            'converged_models': len(convergence_steps),
            'avg_convergence_step': np.mean(convergence_steps) if convergence_steps else None,
            'avg_final_loss': np.mean(final_losses) if final_losses else None,
            'loss_std': np.std(final_losses) if len(final_losses) > 1 else None
        }
    
    def _calculate_confidence_scores(self, ensemble_results: List[Dict]) -> List[float]:
        """Calculate confidence scores for the ensemble."""
        
        confidence_scores = []
        
        for result in ensemble_results:
            base_confidence = 0.5
            
            # Boost for convergence
            if result.get('convergence_step') is not None:
                base_confidence += 0.2
            
            # Boost for low loss
            final_loss = result.get('final_loss', float('inf'))
            if final_loss < 1.0:
                base_confidence += 0.2 * (1.0 - min(final_loss, 1.0))
            
            # Boost for quality scores
            quality_scores = result.get('quality_scores', [])
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                base_confidence += 0.1 * avg_quality
            
            confidence_scores.append(min(1.0, base_confidence))
        
        return confidence_scores


class AdaptiveTaskSolver:
    """Main adaptive task solver orchestrating the entire process."""
    
    def __init__(self, config: SolvingConfig):
        self.config = config
        self.performance_tracker = PerformanceTracker()
        self.ensemble_solver = EnsembleSolver(config)
        self.global_start_time = time.time()
        
    def solve_tasks(self, tasks_data: Dict[str, Any], 
                   task_names: Optional[List[str]] = None) -> Dict[str, TaskResult]:
        """Solve multiple tasks with adaptive resource allocation."""
        
        if task_names is None:
            task_names = list(tasks_data.keys())
        
        results = {}
        total_tasks = len(task_names)
        
        logger.info(f"Starting to solve {total_tasks} tasks with ensemble size {self.config.ensemble_size}")
        
        # Single-threaded processing for better resource control
        for i, task_name in enumerate(task_names):
            logger.info(f"Solving task {i+1}/{total_tasks}: {task_name}")
            
            try:
                # Check time remaining
                elapsed_time = time.time() - self.global_start_time
                remaining_tasks = total_tasks - i
                
                if remaining_tasks > 0:
                    time_per_task = min(
                        self.config.time_limit_per_task,
                        (3600 - elapsed_time) / remaining_tasks  # Assume 1 hour total limit
                    )
                else:
                    time_per_task = self.config.time_limit_per_task
                
                if time_per_task < 30:  # Minimum 30 seconds per task
                    logger.warning(f"Low time budget: {time_per_task:.1f}s for {task_name}")
                
                # Solve task
                task_data = tasks_data[task_name]
                result = self.ensemble_solver.solve_with_ensemble(task_name, task_data)
                
                # Update performance tracking
                self.performance_tracker.update(task_name, result)
                
                results[task_name] = result
                
                logger.info(f"Completed {task_name}: "
                          f"success={result.success}, "
                          f"time={result.solving_time:.1f}s, "
                          f"memory={result.peak_memory_mb:.1f}MB")
                
            except Exception as e:
                error_msg = traceback.format_exc()
                logger.error(f"Failed to solve {task_name}: {error_msg}")
                
                results[task_name] = TaskResult(
                    task_name=task_name,
                    solutions=self._create_emergency_fallback(task_data),
                    metadata={'error': str(e)},
                    success=False,
                    solving_time=0.0,
                    peak_memory_mb=0.0,
                    iterations_completed=0,
                    error_message=error_msg
                )
        
        self._log_final_statistics(results)
        return results
    
    def _create_emergency_fallback(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create emergency fallback when everything fails."""
        n_test = len(task_data.get('test', []))
        return [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in range(n_test)]
    
    def _log_final_statistics(self, results: Dict[str, TaskResult]):
        """Log comprehensive final statistics."""
        
        successful_tasks = sum(1 for r in results.values() if r.success)
        total_tasks = len(results)
        
        total_time = sum(r.solving_time for r in results.values())
        avg_memory = np.mean([r.peak_memory_mb for r in results.values()])
        
        logger.info(f"=== FINAL STATISTICS ===")
        logger.info(f"Tasks solved: {successful_tasks}/{total_tasks} ({100*successful_tasks/total_tasks:.1f}%)")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average memory: {avg_memory:.1f}MB")
        logger.info(f"Average time per task: {total_time/total_tasks:.1f}s")


def solve_arc_competition(challenges_path: str, 
                         output_path: str = "submission.json",
                         config: Optional[SolvingConfig] = None) -> Dict[str, Any]:
    """
    Main entry point for solving ARC competition.
    
    Args:
        challenges_path: Path to challenges JSON file
        output_path: Path for output submission file
        config: Solving configuration (uses default if None)
    
    Returns:
        Dictionary containing solving results and metadata
    """
    
    if config is None:
        config = SolvingConfig(
            time_limit_per_task=300.0,
            n_train_iterations=2000,
            use_ensemble=True,
            ensemble_size=5,
            adaptive_iterations=True,
            use_mixed_precision=True
        )
    
    # Load challenges
    logger.info(f"Loading challenges from {challenges_path}")
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    # Initialize solver
    solver = AdaptiveTaskSolver(config)
    
    # Solve all tasks
    results = solver.solve_tasks(challenges)
    
    # Create submission format
    submission = {}
    metadata = {
        'config': asdict(config),
        'performance_summary': {},
        'task_metadata': {}
    }
    
    for task_name, result in results.items():
        submission[task_name] = result.solutions
        metadata['task_metadata'][task_name] = result.metadata
    
    # Add performance summary
    successful_tasks = sum(1 for r in results.values() if r.success)
    metadata['performance_summary'] = {
        'total_tasks': len(results),
        'successful_tasks': successful_tasks,
        'success_rate': successful_tasks / len(results),
        'total_solving_time': sum(r.solving_time for r in results.values()),
        'average_memory_mb': np.mean([r.peak_memory_mb for r in results.values()])
    }
    
    # Save submission
    logger.info(f"Saving submission to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # Save metadata
    metadata_path = output_path.replace('.json', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"=== COMPETITION SOLVING COMPLETE ===")
    logger.info(f"Submission saved: {output_path}")
    logger.info(f"Metadata saved: {metadata_path}")
    logger.info(f"Success rate: {100*successful_tasks/len(results):.1f}%")
    
    return {
        'submission': submission,
        'metadata': metadata,
        'results': results
    }


# Convenience function for Kaggle notebooks
def solve_task_for_kaggle(task_name: str, split: str, time_limit: float, 
                         n_train_iterations: int, gpu_id: int = 0) -> List[Dict[str, Any]]:
    """
    Simplified interface for Kaggle notebook integration.
    
    Args:
        task_name: Name of the task to solve
        split: Data split ('test', 'evaluation', etc.)
        time_limit: Maximum time for solving (ignored, managed internally)
        n_train_iterations: Number of training iterations
        gpu_id: GPU device ID
    
    Returns:
        List of solution dictionaries in Kaggle format
    """
    
    try:
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = f'cuda:{gpu_id}'
        else:
            device = 'cpu'
        
        torch.set_default_device(device)
        
        # Load task data
        data_path = f'/kaggle/input/arc-prize-2025/arc-agi_{split}_challenges.json'
        with open(data_path, 'r') as f:
            challenges = json.load(f)
        
        if task_name not in challenges:
            logger.error(f"Task {task_name} not found in {split} split")
            return [{'attempt_1': [[0]], 'attempt_2': [[0]]}]
        
        task_data = challenges[task_name]
        
        # Create optimized config for Kaggle
        config = SolvingConfig(
            time_limit_per_task=min(time_limit, 300.0),
            n_train_iterations=min(n_train_iterations, 2000),
            use_ensemble=True,
            ensemble_size=3,  # Reduced for speed
            adaptive_iterations=True,
            save_visualizations=False,
            memory_limit_gb=6.0,  # Conservative for Kaggle
            use_mixed_precision=True,
            early_stopping_patience=100,
            warmup_iterations=50
        )
        
        # Solve single task
        solver = AdaptiveTaskSolver(config)
        ensemble_solver = EnsembleSolver(config)
        
        result = ensemble_solver.solve_with_ensemble(task_name, task_data)
        
        if result.success:
            logger.info(f"Successfully solved {task_name} in {result.solving_time:.1f}s")
            return result.solutions
        else:
            logger.warning(f"Failed to solve {task_name}: {result.error_message}")
            return ensemble_solver._create_fallback_solutions(task_data)
    
    except Exception as e:
        logger.error(f"Kaggle solver failed for {task_name}: {e}")
        return [{'attempt_1': [[0]], 'attempt_2': [[0]]}]


def solve_all_tasks_kaggle(split: str = 'test', time_limit_per_task: float = 300.0,
                          n_train_iterations: int = 2000) -> Dict[str, List[Dict[str, Any]]]:
    """
    Solve all tasks in Kaggle format.
    
    Args:
        split: Data split to solve
        time_limit_per_task: Time limit per task
        n_train_iterations: Training iterations per task
    
    Returns:
        Dictionary mapping task names to solutions
    """
    
    try:
        # Load all challenges
        data_path = f'/kaggle/input/arc-prize-2025/arc-agi_{split}_challenges.json'
        with open(data_path, 'r') as f:
            challenges = json.load(f)
        
        # Create config
        config = SolvingConfig(
            time_limit_per_task=time_limit_per_task,
            n_train_iterations=n_train_iterations,
            use_ensemble=True,
            ensemble_size=3,
            adaptive_iterations=True,
            memory_limit_gb=6.0,
            use_mixed_precision=True
        )
        
        # Solve all tasks
        solver = AdaptiveTaskSolver(config)
        results = solver.solve_tasks(challenges)
        
        # Convert to Kaggle format
        kaggle_format = {}
        for task_name, result in results.items():
            kaggle_format[task_name] = result.solutions
        
        return kaggle_format
    
    except Exception as e:
        logger.error(f"Failed to solve all tasks: {e}")
        return {}


class CompetitionOptimizer:
    """Advanced optimizer for competition-specific improvements."""
    
    def __init__(self):
        self.solution_cache = {}
        self.pattern_library = {}
        self.success_patterns = defaultdict(list)
    
    def optimize_submission(self, submission: Dict[str, Any], 
                          challenges: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply competition-specific optimizations to submission.
        
        Args:
            submission: Raw submission dictionary
            challenges: Challenge data for pattern analysis
        
        Returns:
            Optimized submission
        """
        
        optimized = {}
        
        for task_name, solutions in submission.items():
            try:
                challenge = challenges.get(task_name, {})
                optimized_solutions = self._optimize_task_solutions(
                    solutions, challenge, task_name
                )
                optimized[task_name] = optimized_solutions
                
            except Exception as e:
                logger.warning(f"Failed to optimize {task_name}: {e}")
                optimized[task_name] = solutions
        
        return optimized
    
    def _optimize_task_solutions(self, solutions: List[Dict], challenge: Dict, 
                               task_name: str) -> List[Dict]:
        """Optimize solutions for a specific task."""
        
        optimized_solutions = []
        
        for sol_idx, solution in enumerate(solutions):
            try:
                # Apply various optimization strategies
                optimized_sol = self._apply_size_optimization(solution, challenge)
                optimized_sol = self._apply_pattern_optimization(optimized_sol, challenge)
                optimized_sol = self._apply_symmetry_optimization(optimized_sol, challenge)
                optimized_sol = self._apply_color_optimization(optimized_sol, challenge)
                
                optimized_solutions.append(optimized_sol)
                
            except Exception as e:
                logger.warning(f"Failed to optimize solution {sol_idx} for {task_name}: {e}")
                optimized_solutions.append(solution)
        
        return optimized_solutions
    
    def _apply_size_optimization(self, solution: Dict, challenge: Dict) -> Dict:
        """Optimize grid sizes based on training examples."""
        
        try:
            train_examples = challenge.get('train', [])
            if not train_examples:
                return solution
            
            # Analyze size patterns
            input_sizes = []
            output_sizes = []
            
            for example in train_examples:
                if 'input' in example and 'output' in example:
                    input_grid = example['input']
                    output_grid = example['output']
                    
                    input_sizes.append((len(input_grid), len(input_grid[0])))
                    output_sizes.append((len(output_grid), len(output_grid[0])))
            
            # Check for consistent patterns
            if len(set(output_sizes)) == 1:
                # All outputs same size
                target_h, target_w = output_sizes[0]
                
                for attempt_key in ['attempt_1', 'attempt_2']:
                    if attempt_key in solution:
                        solution[attempt_key] = self._resize_grid(
                            solution[attempt_key], target_h, target_w
                        )
            
            elif len(set(input_sizes)) == len(set(output_sizes)) == 1:
                # Input-output size relationship
                input_h, input_w = input_sizes[0]
                output_h, output_w = output_sizes[0]
                
                # Apply same scaling
                for attempt_key in ['attempt_1', 'attempt_2']:
                    if attempt_key in solution:
                        current_grid = solution[attempt_key]
                        if current_grid:
                            current_h, current_w = len(current_grid), len(current_grid[0])
                            
                            # Scale proportionally
                            scale_h = output_h / input_h if input_h > 0 else 1
                            scale_w = output_w / input_w if input_w > 0 else 1
                            
                            new_h = max(1, int(current_h * scale_h))
                            new_w = max(1, int(current_w * scale_w))
                            
                            solution[attempt_key] = self._resize_grid(
                                current_grid, new_h, new_w
                            )
            
            return solution
            
        except Exception:
            return solution
    
    def _resize_grid(self, grid: List[List[int]], target_h: int, target_w: int) -> List[List[int]]:
        """Resize grid to target dimensions."""
        
        if not grid or target_h <= 0 or target_w <= 0:
            return [[0 for _ in range(max(1, target_w))] for _ in range(max(1, target_h))]
        
        current_h, current_w = len(grid), len(grid[0])
        
        if current_h == target_h and current_w == target_w:
            return grid
        
        # Create new grid
        new_grid = []
        
        for i in range(target_h):
            new_row = []
            
            # Map to source row
            src_i = min(i, current_h - 1) if current_h > 0 else 0
            
            for j in range(target_w):
                # Map to source column
                src_j = min(j, current_w - 1) if current_w > 0 else 0
                
                if src_i < len(grid) and src_j < len(grid[src_i]):
                    new_row.append(grid[src_i][src_j])
                else:
                    new_row.append(0)
            
            new_grid.append(new_row)
        
        return new_grid
    
    def _apply_pattern_optimization(self, solution: Dict, challenge: Dict) -> Dict:
        """Apply pattern-based optimizations."""
        
        try:
            train_examples = challenge.get('train', [])
            if len(train_examples) < 2:
                return solution
            
            # Detect common transformations
            transformations = []
            
            for example in train_examples:
                if 'input' in example and 'output' in example:
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])
                    
                    # Detect transformation type
                    if input_grid.shape == output_grid.shape:
                        # Same size transformations
                        diff_mask = input_grid != output_grid
                        transformation = {
                            'type': 'cell_modification',
                            'changes': np.sum(diff_mask),
                            'change_ratio': np.sum(diff_mask) / input_grid.size
                        }
                        transformations.append(transformation)
            
            # Apply detected patterns to solutions
            if transformations:
                avg_change_ratio = np.mean([t['change_ratio'] for t in transformations])
                
                # If very few changes, prefer simpler solutions
                if avg_change_ratio < 0.1:
                    for attempt_key in ['attempt_1', 'attempt_2']:
                        if attempt_key in solution:
                            solution[attempt_key] = self._simplify_grid(
                                solution[attempt_key]
                            )
            
            return solution
            
        except Exception:
            return solution
    
    def _apply_symmetry_optimization(self, solution: Dict, challenge: Dict) -> Dict:
        """Apply symmetry-based optimizations."""
        
        try:
            train_examples = challenge.get('train', [])
            
            # Check for symmetry patterns
            symmetry_patterns = []
            
            for example in train_examples:
                if 'output' in example:
                    output_grid = np.array(example['output'])
                    
                    # Check various symmetries
                    is_h_symmetric = np.array_equal(output_grid, np.fliplr(output_grid))
                    is_v_symmetric = np.array_equal(output_grid, np.flipud(output_grid))
                    is_diagonal_symmetric = (output_grid.shape[0] == output_grid.shape[1] and 
                                           np.array_equal(output_grid, output_grid.T))
                    
                    symmetry_patterns.append({
                        'horizontal': is_h_symmetric,
                        'vertical': is_v_symmetric,
                        'diagonal': is_diagonal_symmetric
                    })
            
            # Apply symmetry if consistent
            if len(symmetry_patterns) >= 2:
                h_symmetric = all(p['horizontal'] for p in symmetry_patterns)
                v_symmetric = all(p['vertical'] for p in symmetry_patterns)
                d_symmetric = all(p['diagonal'] for p in symmetry_patterns)
                
                for attempt_key in ['attempt_1', 'attempt_2']:
                    if attempt_key in solution:
                        grid = np.array(solution[attempt_key])
                        
                        if h_symmetric and grid.shape[1] > 1:
                            # Enforce horizontal symmetry
                            for i in range(grid.shape[0]):
                                for j in range(grid.shape[1] // 2):
                                    mirror_j = grid.shape[1] - 1 - j
                                    # Average the symmetric positions
                                    avg_val = (grid[i, j] + grid[i, mirror_j]) // 2
                                    grid[i, j] = avg_val
                                    grid[i, mirror_j] = avg_val
                        
                        if v_symmetric and grid.shape[0] > 1:
                            # Enforce vertical symmetry
                            for i in range(grid.shape[0] // 2):
                                for j in range(grid.shape[1]):
                                    mirror_i = grid.shape[0] - 1 - i
                                    avg_val = (grid[i, j] + grid[mirror_i, j]) // 2
                                    grid[i, j] = avg_val
                                    grid[mirror_i, j] = avg_val
                        
                        solution[attempt_key] = grid.tolist()
            
            return solution
            
        except Exception:
            return solution
    
    def _apply_color_optimization(self, solution: Dict, challenge: Dict) -> Dict:
        """Optimize color usage based on training examples."""
        
        try:
            train_examples = challenge.get('train', [])
            
            # Collect color usage statistics
            color_stats = defaultdict(int)
            
            for example in train_examples:
                for grid_type in ['input', 'output']:
                    if grid_type in example:
                        grid = example[grid_type]
                        for row in grid:
                            for cell in row:
                                color_stats[cell] += 1
            
            # Get valid colors (those that appear in training)
            valid_colors = set(color_stats.keys())
            
            # Clean up solutions to use only valid colors
            for attempt_key in ['attempt_1', 'attempt_2']:
                if attempt_key in solution:
                    grid = solution[attempt_key]
                    cleaned_grid = []
                    
                    for row in grid:
                        cleaned_row = []
                        for cell in row:
                            if cell in valid_colors:
                                cleaned_row.append(cell)
                            else:
                                # Replace with most common valid color
                                most_common_color = max(valid_colors, 
                                                      key=lambda c: color_stats[c])
                                cleaned_row.append(most_common_color)
                        cleaned_grid.append(cleaned_row)
                    
                    solution[attempt_key] = cleaned_grid
            
            return solution
            
        except Exception:
            return solution
    
    def _simplify_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Simplify grid by reducing noise."""
        
        try:
            if not grid or len(grid) < 3 or len(grid[0]) < 3:
                return grid
            
            grid_array = np.array(grid)
            simplified = grid_array.copy()
            
            # Apply median filter to reduce noise
            for i in range(1, grid_array.shape[0] - 1):
                for j in range(1, grid_array.shape[1] - 1):
                    neighborhood = grid_array[i-1:i+2, j-1:j+2]
                    simplified[i, j] = np.median(neighborhood)
            
            return simplified.tolist()
            
        except Exception:
            return grid


# Export main functions for easy import
__all__ = [
    'SolvingConfig',
    'TaskResult', 
    'AdaptiveTaskSolver',
    'EnsembleSolver',
    'solve_arc_competition',
    'solve_task_for_kaggle',
    'solve_all_tasks_kaggle',
    'CompetitionOptimizer'
]