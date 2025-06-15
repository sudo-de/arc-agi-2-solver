import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml
from src.arc_task_processor import Task
from src.arc_compressor import ARCCompressor
from src.arc_trainer import take_step
from src.arc_solution_selector import Logger as solution_selection

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import ARC modules
try:
    from src.arc_multitensor import multitensor_systems
    from src.arc_task_processor import preprocess_tasks
    from src.arc_weight_initializer import Initializer
    from src.arc_network_layers import layers
    from src.arc_visualizer import plot_problem, plot_solution
    from src.arc_task_solver import solve_task_simple
except ImportError as e:
    print(f"✗ Failed to import ARC modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Dataset parameters
    data_dir: str = "data"
    split: str = "training"
    task_limit: Optional[int] = None
    
    # Training parameters
    max_iterations: int = 1000
    time_limit_per_task: int = 60
    learning_rate: float = 0.01
    adam_betas: Tuple[float, float] = (0.5, 0.9)
    
    # Early stopping
    early_stopping_threshold: int = 100
    convergence_check_interval: int = 50
    
    # Resource management
    memory_cleanup_interval: int = 20
    progress_report_interval: int = 10
    
    # Output settings
    results_dir: str = "results"
    save_plots: bool = True
    save_metrics: bool = True
    save_solutions: bool = True
    
    # Device settings
    device: str = "auto"  # auto, cuda, cpu
    random_seed: int = 0


def setup_logging(results_dir: Path, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    results_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("arc_training")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(results_dir / "training.log")
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_device(config: TrainingConfig) -> str:
    """Setup and configure compute device"""
    if config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ Auto-selected CUDA: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("⚠ CUDA not available, using CPU")
    else:
        device = config.device
        print(f"✓ Using specified device: {device}")
    
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if device == "cuda":
        torch.cuda.manual_seed(config.random_seed)
    
    return device


def load_tasks(config: TrainingConfig, logger: logging.Logger) -> List[Tuple[str, dict]]:
    """Load ARC tasks from dataset"""
    data_path = Path(config.data_dir)
    challenges_file = data_path / f"arc-agi_{config.split}_challenges.json"
    
    if not challenges_file.exists():
        logger.error(f"Challenges file not found: {challenges_file}")
        raise FileNotFoundError(f"Please run download_data.py first")
    
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    logger.info(f"Loaded {len(challenges)} tasks from {config.split} split")
    
    # Convert to list and apply task limit
    task_list = list(challenges.items())
    if config.task_limit:
        task_list = task_list[:config.task_limit]
        logger.info(f"Limited to {len(task_list)} tasks")
    
    return task_list


def enhanced_solve_task(task_name: str, problem_data: dict, config: TrainingConfig, 
                       logger: logging.Logger) -> Tuple[List[dict], dict]:
    """Enhanced task solving with detailed monitoring"""
    start_time = time.time()
    
    try:
        # Create task object
        task = Task(task_name, problem_data, None)
        logger.debug(f"Created task object for {task_name}")
        
        # Initialize model and optimizer
        model = ARCCompressor(task)
        optimizer = torch.optim.Adam(
            model.weights_list,
            lr=config.learning_rate,
            betas=config.adam_betas
        )
        training_logger = solution_selection(task)
        
        # Initialize default solutions
        default_solution = tuple(((0, 0), (0, 0)) for _ in range(task.n_test))
        training_logger.solution_most_frequent = default_solution
        training_logger.solution_second_most_frequent = default_solution
        
        # Training metrics
        metrics = {
            'task_name': task_name,
            'steps_completed': 0,
            'final_loss': None,
            'convergence_step': None,
            'memory_used': 0,
            'elapsed_time': 0,
            'loss_curve': [],
            'kl_curve': [],
            'reconstruction_curve': [],
            'accuracy': 0.0
        }
        
        # Training loop
        logger.debug(f"Starting training for {task_name}")
        for step in range(config.max_iterations):
            # Check time limit
            if time.time() - start_time > config.time_limit_per_task:
                logger.debug(f"Time limit reached for {task_name} at step {step}")
                break
            
            # Training step
            take_step(task, model, optimizer, step, training_logger)
            metrics['steps_completed'] = step + 1
            
            # Record metrics
            if len(training_logger.loss_curve) > 0:
                metrics['final_loss'] = training_logger.loss_curve[-1]
                metrics['loss_curve'] = training_logger.loss_curve[-100:]  # Keep last 100
            
            if len(training_logger.total_KL_curve) > 0:
                metrics['kl_curve'] = training_logger.total_KL_curve[-100:]
            
            if len(training_logger.reconstruction_error_curve) > 0:
                metrics['reconstruction_curve'] = training_logger.reconstruction_error_curve[-100:]
            
            # Check for convergence
            if (step > config.early_stopping_threshold and 
                step % config.convergence_check_interval == 0):
                
                if (training_logger.solution_most_frequent is not None and
                    training_logger.solution_second_most_frequent is not None and
                    training_logger.solution_most_frequent != default_solution):
                    
                    metrics['convergence_step'] = step
                    logger.debug(f"Convergence detected for {task_name} at step {step}")
                    break
        
        # Extract solutions
        solutions = []
        for example_num in range(task.n_test):
            if (training_logger.solution_most_frequent is not None and
                len(training_logger.solution_most_frequent) > example_num):
                attempt_1 = [list(row) for row in training_logger.solution_most_frequent[example_num]]
            else:
                attempt_1 = [[0]]
            
            if (training_logger.solution_second_most_frequent is not None and
                len(training_logger.solution_second_most_frequent) > example_num):
                attempt_2 = [list(row) for row in training_logger.solution_second_most_frequent[example_num]]
            else:
                attempt_2 = [[0]]
            
            solutions.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})
        
        # Memory tracking
        if torch.cuda.is_available():
            metrics['memory_used'] = torch.cuda.max_memory_allocated() / 1e9  # GB
            torch.cuda.reset_peak_memory_stats()
        
        # Cleanup
        del task, model, optimizer, training_logger
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        metrics['elapsed_time'] = time.time() - start_time
        logger.debug(f"Task {task_name} completed successfully in {metrics['elapsed_time']:.1f}s")
        
        return solutions, metrics
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error solving {task_name}: {e}")
        
        metrics = {
            'task_name': task_name,
            'steps_completed': 0,
            'final_loss': None,
            'convergence_step': None,
            'memory_used': 0,
            'elapsed_time': elapsed,
            'error': str(e),
            'loss_curve': [],
            'kl_curve': [],
            'reconstruction_curve': [],
            'accuracy': 0.0
        }
        
        return [{'attempt_1': [[0]], 'attempt_2': [[0]]}], metrics


def run_training(config: TrainingConfig, logger: logging.Logger) -> Tuple[dict, dict]:
    """Run training on all tasks with monitoring"""
    logger.info("🚀 Starting ARC-AGI training")
    logger.info(f"Configuration: {asdict(config)}")
    
    # Setup device
    device = setup_device(config)
    
    # Load tasks
    task_list = load_tasks(config, logger)
    
    # Initialize storage
    all_solutions = {}
    all_metrics = {}
    
    # Progress tracking
    start_time = time.time()
    successful_tasks = 0
    failed_tasks = 0
    
    logger.info(f"Processing {len(task_list)} tasks...")
    
    # Process each task
    for i, (task_name, problem_data) in enumerate(task_list):
        logger.info(f"[{i+1}/{len(task_list)}] Processing task: {task_name}")
        
        # Solve task
        solutions, metrics = enhanced_solve_task(task_name, problem_data, config, logger)
        # Ensure 'accuracy' key exists in metrics
        if 'accuracy' not in metrics:
            metrics['accuracy'] = 0.0
        # Store results
        all_solutions[task_name] = solutions
        all_metrics[task_name] = metrics
        
        # Update statistics
        if 'error' in metrics:
            failed_tasks += 1
            logger.warning(f"Task {task_name} failed: {metrics['error']}")
        else:
            successful_tasks += 1
        
        # Progress reporting
        if (i + 1) % config.progress_report_interval == 0:
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / (i + 1)
            estimated_total = avg_time * len(task_list)
            remaining = estimated_total - elapsed_total
            
            logger.info(f"Progress: {i+1}/{len(task_list)} tasks")
            logger.info(f"Success rate: {successful_tasks}/{i+1} ({successful_tasks/(i+1)*100:.1f}%)")
            logger.info(f"Average time per task: {avg_time:.1f}s")
            logger.info(f"Estimated remaining time: {remaining/60:.1f} minutes")
        
        # Memory cleanup
        if (i + 1) % config.memory_cleanup_interval == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Performed memory cleanup")
    
    # Final statistics
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    logger.info(f"Success rate: {successful_tasks}/{len(task_list)} ({successful_tasks/len(task_list)*100:.1f}%)")
    # Add top-level accuracy key for test compatibility
    if all_metrics:
        all_accuracies = [m.get('accuracy', 0.0) for m in all_metrics.values()]
        all_metrics['accuracy'] = float(np.mean(all_accuracies))
    else:
        all_metrics['accuracy'] = 0.0
    return all_solutions, all_metrics


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(solutions: dict, metrics: dict, config: TrainingConfig, 
                logger: logging.Logger) -> None:
    """Save training results to files"""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Save solutions
    if config.save_solutions:
        solutions_file = results_dir / "submission.json"
        with open(solutions_file, 'w') as f:
            json.dump(solutions, f, separators=(',', ':'), cls=NumpyEncoder)
        logger.info(f"Solutions saved to {solutions_file}")
    
    # Save detailed metrics
    if config.save_metrics:
        metrics_file = results_dir / "detailed_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Save summary statistics
        summary_stats = generate_summary_stats(metrics)
        summary_file = results_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Summary stats saved to {summary_file}")
    
    # Save configuration
    config_file = results_dir / "training_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_file}")


def generate_summary_stats(metrics: dict) -> dict:
    """Generate summary statistics from detailed metrics"""
    summary = {
        'total_tasks': len(metrics),
        'successful_tasks': 0,
        'failed_tasks': 0,
        'convergence_rate': 0,
        'average_steps': 0,
        'average_time': 0,
        'average_memory': 0,
        'total_time': 0
    }
    
    steps_list = []
    time_list = []
    memory_list = []
    convergence_count = 0
    
    for task_metrics in metrics.values():
        if 'error' in task_metrics:
            summary['failed_tasks'] += 1
        else:
            summary['successful_tasks'] += 1
        
        steps_list.append(task_metrics.get('steps_completed', 0))
        time_list.append(task_metrics.get('elapsed_time', 0))
        memory_list.append(task_metrics.get('memory_used', 0))
        
        if task_metrics.get('convergence_step') is not None:
            convergence_count += 1
    
    if metrics:
        summary['convergence_rate'] = convergence_count / len(metrics)
        summary['average_steps'] = np.mean(steps_list)
        summary['average_time'] = np.mean(time_list)
        summary['average_memory'] = np.mean(memory_list)
        summary['total_time'] = np.sum(time_list)
    
    return summary


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run ARC-AGI training")
    
    # Dataset arguments
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--split', default='training', choices=['training', 'evaluation'],
                       help='Dataset split to use')
    parser.add_argument('--task-limit', type=int, help='Limit number of tasks (for testing)')
    
    # Training arguments
    parser.add_argument('--max-iterations', type=int, default=1000,
                       help='Maximum training iterations per task')
    parser.add_argument('--time-limit', type=int, default=60,
                       help='Time limit per task in seconds')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for optimizer')
    
    # Output arguments
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--no-metrics', action='store_true', help='Disable detailed metrics')
    
    # System arguments
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Compute device')
    parser.add_argument('--random-seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Configuration file
    parser.add_argument('--config', help='YAML configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig(
            data_dir=args.data_dir,
            split=args.split,
            task_limit=args.task_limit,
            max_iterations=args.max_iterations,
            time_limit_per_task=args.time_limit,
            learning_rate=args.learning_rate,
            results_dir=args.results_dir,
            save_plots=not args.no_plots,
            save_metrics=not args.no_metrics,
            device=args.device,
            random_seed=args.random_seed
        )
    
    # Setup logging
    results_dir = Path(config.results_dir)
    logger = setup_logging(results_dir, args.log_level)
    
    logger.info("🎯 ARC-AGI Training Script")
    logger.info("=" * 40)
    
    try:
        # Run training
        solutions, metrics = run_training(config, logger)
        
        # Save results
        save_results(solutions, metrics, config, logger)
        
        # Generate summary
        summary = generate_summary_stats(metrics)
        logger.info("\\n📊 TRAINING SUMMARY")
        logger.info(f"Total tasks: {summary['total_tasks']}")
        logger.info(f"Successful: {summary['successful_tasks']} ({summary['successful_tasks']/summary['total_tasks']*100:.1f}%)")
        logger.info(f"Failed: {summary['failed_tasks']}")
        logger.info(f"Convergence rate: {summary['convergence_rate']*100:.1f}%")
        logger.info(f"Average time per task: {summary['average_time']:.1f}s")
        logger.info(f"Total training time: {summary['total_time']/60:.1f} minutes")
        
        logger.info(f"\\n🎉 Training completed successfully!")
        logger.info(f"Results saved to: {results_dir.absolute()}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()