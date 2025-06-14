import sys
import argparse
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.arc_task_solver import solve_multiple_tasks, create_kaggle_submission

def generate_kaggle_submission(
    data_dir: str = "data",
    output_file: str = "submission.json",
    time_limit: float = 300.0,
    iterations: int = 2000,
    max_workers: int = None
) -> None:
    """Generate final Kaggle submission."""
    
    print("Generating Kaggle submission...")
    
    # Load test tasks
    test_file = Path(data_dir) / "raw" / "arc-agi_test_challenges.json"
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    task_names = list(test_data.keys())
    print(f"Loaded {len(task_names)} test tasks")
    
    # Estimate timing
    if max_workers is None:
        import torch
        max_workers = min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    
    estimated_time = (len(task_names) * time_limit) / max_workers
    print(f"Estimated completion time: {estimated_time/60:.1f} minutes")
    print(f"Using {max_workers} workers")
    
    # Solve all tasks
    start_time = time.time()
    
    results = solve_multiple_tasks(
        task_names=task_names,
        split='test',
        time_limit_per_task=time_limit,
        n_train_iterations=iterations,
        max_workers=max_workers,
        data_path=str(Path(data_dir) / "raw")
    )
    
    total_time = time.time() - start_time
    
    # Create submission
    create_kaggle_submission(results, output_file)
    
    # Results summary
    successful = sum(1 for r in results.values() if 'solutions' in r)
    print(f"\nSubmission generation completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful tasks: {successful}/{len(task_names)}")
    print(f"Success rate: {successful/len(task_names)*100:.1f}%")
    print(f"Submission saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output", default="submission.json", help="Output submission file")
    parser.add_argument("--time-limit", type=float, default=300.0, help="Time limit per task")
    parser.add_argument("--iterations", type=int, default=2000, help="Training iterations")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    
    args = parser.parse_args()
    
    generate_kaggle_submission(
        data_dir=args.data_dir,
        output_file=args.output,
        time_limit=args.time_limit,
        iterations=args.iterations,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()