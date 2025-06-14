import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.arc_task_solver import solve_multiple_tasks, create_kaggle_submission
import json

def load_task_names(data_dir: str, split: str) -> list:
    """Load task names from the specified split."""
    data_path = Path(data_dir) / "raw" / f"arc-agi_{split}_challenges.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return list(data.keys())

def run_training_experiment(
    split: str = "training",
    max_tasks: int = 5,
    time_limit: float = 120.0,
    iterations: int = 500,
    data_dir: str = "data"
) -> None:
    """Run a training experiment on a subset of tasks."""
    
    print(f"Starting training experiment on {split} split")
    print(f"Max tasks: {max_tasks}, Time limit: {time_limit}s, Iterations: {iterations}")
    
    # Load task names
    try:
        task_names = load_task_names(data_dir, split)
        if max_tasks:
            task_names = task_names[:max_tasks]
        
        print(f"Loaded {len(task_names)} tasks: {task_names}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python scripts/download_data.py' first")
        return
    
    # Run training
    start_time = time.time()
    
    results = solve_multiple_tasks(
        task_names=task_names,
        split=split,
        time_limit_per_task=time_limit,
        n_train_iterations=iterations,
        max_workers=1,  # Single worker for debugging
        data_path=str(Path(data_dir) / "raw")
    )
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results.values() if 'solutions' in r)
    print(f"\nExperiment completed in {total_time:.1f} seconds")
    print(f"Successful tasks: {successful}/{len(task_names)}")
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create submission file
    submission_file = output_dir / f"experiment_{split}_{max_tasks}tasks.json"
    create_kaggle_submission(results, str(submission_file))
    
    # Save detailed results
    results_file = output_dir / f"detailed_results_{split}_{max_tasks}tasks.json"
    with open(results_file, 'w') as f:
        # Convert any non-serializable objects
        serializable_results = {}
        for task_name, result in results.items():
            serializable_results[task_name] = {
                'solutions': result.get('solutions', []),
                'memory_used': result.get('memory_used', 0),
                'iterations_completed': result.get('iterations_completed', 0)
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    print(f"Submission saved to {submission_file}")

def main():
    parser = argparse.ArgumentParser(description="Run ARC training experiment")
    parser.add_argument("--split", default="training", choices=["training", "evaluation", "test"])
    parser.add_argument("--max-tasks", type=int, default=5, help="Maximum number of tasks")
    parser.add_argument("--time-limit", type=float, default=120.0, help="Time limit per task")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    
    args = parser.parse_args()
    
    run_training_experiment(
        split=args.split,
        max_tasks=args.max_tasks,
        time_limit=args.time_limit,
        iterations=args.iterations,
        data_dir=args.data_dir
    )

if __name__ == "__main__":
    main()