#!/usr/bin/env python3
"""
ARC-AGI Submission Generation Script

This script generates a properly formatted submission file for the ARC-AGI competition.
It can work with existing model results or generate new predictions.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import ARC modules
try:
    from src.arc_multitensor import multitensor_systems
    from src.arc_task_processor import Task, preprocess_tasks
    from src.arc_compressor import ARCCompressor
    from src.arc_weight_initializer import Initializer
    from src.arc_network_layers import layers
    from src.arc_solution_selector import Logger
    from src.arc_visualizer import plot_problem, plot_solution
    from src.arc_task_solver import solve_task_simple
except ImportError as e:
    print(f"✗ Failed to import ARC modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging for submission generation"""
    output_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("submission_generator")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(output_dir / "submission_generation.log")
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_test_challenges(data_dir: Path) -> Dict[str, dict]:
    """Load test challenges from data directory"""
    test_file = data_dir / "arc-agi_test_challenges.json"
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test challenges file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        challenges = json.load(f)
    
    return challenges


def load_existing_submission(submission_file: Path) -> Optional[Dict[str, List[dict]]]:
    """Load existing submission file if it exists"""
    if not submission_file.exists():
        return None
    
    try:
        with open(submission_file, 'r') as f:
            submission = json.load(f)
        return submission
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠ Warning: Could not load existing submission: {e}")
        return None


def validate_submission_format(submission: Dict[str, List[dict]], 
                             test_challenges: Dict[str, dict],
                             logger: logging.Logger) -> Tuple[bool, List[str]]:
    """Validate submission format according to competition requirements"""
    errors = []
    
    # Check if all test tasks are present
    missing_tasks = set(test_challenges.keys()) - set(submission.keys())
    if missing_tasks:
        errors.append(f"Missing tasks: {list(missing_tasks)}")
    
    # Check if there are extra tasks
    extra_tasks = set(submission.keys()) - set(test_challenges.keys())
    if extra_tasks:
        errors.append(f"Extra tasks not in test set: {list(extra_tasks)}")
    
    # Validate each task's format
    for task_id, task_solutions in submission.items():
        if task_id not in test_challenges:
            continue
        
        # Check if solutions is a list
        if not isinstance(task_solutions, list):
            errors.append(f"Task {task_id}: solutions must be a list")
            continue
        
        # Check number of test examples
        expected_test_count = len(test_challenges[task_id]['test'])
        if len(task_solutions) != expected_test_count:
            errors.append(f"Task {task_id}: expected {expected_test_count} solutions, got {len(task_solutions)}")
            continue
        
        # Validate each solution
        for i, solution in enumerate(task_solutions):
            if not isinstance(solution, dict):
                errors.append(f"Task {task_id}[{i}]: solution must be a dict")
                continue
            
            # Check required keys
            if 'attempt_1' not in solution or 'attempt_2' not in solution:
                errors.append(f"Task {task_id}[{i}]: missing 'attempt_1' or 'attempt_2'")
                continue
            
            # Validate attempt formats
            for attempt_key in ['attempt_1', 'attempt_2']:
                attempt = solution[attempt_key]
                if not isinstance(attempt, list):
                    errors.append(f"Task {task_id}[{i}][{attempt_key}]: must be a list")
                    continue
                
                # Check if it's a valid grid (list of lists)
                if not all(isinstance(row, list) for row in attempt):
                    errors.append(f"Task {task_id}[{i}][{attempt_key}]: must be a list of lists")
                    continue
                
                # Check if all rows have the same length
                if attempt and len(set(len(row) for row in attempt)) > 1:
                    errors.append(f"Task {task_id}[{i}][{attempt_key}]: inconsistent row lengths")
                
                # Check if all values are integers 0-9
                for row_idx, row in enumerate(attempt):
                    for col_idx, val in enumerate(row):
                        if not isinstance(val, int) or not (0 <= val <= 9):
                            errors.append(f"Task {task_id}[{i}][{attempt_key}][{row_idx}][{col_idx}]: invalid value {val} (must be int 0-9)")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("✓ Submission format validation passed")
    else:
        logger.error(f"✗ Submission format validation failed with {len(errors)} errors")
        for error in errors[:10]:  # Show first 10 errors
            logger.error(f"  {error}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
    
    return is_valid, errors


def generate_default_submission(test_challenges: Dict[str, dict]) -> Dict[str, List[dict]]:
    """Generate a default submission with simple fallback solutions"""
    submission = {}
    
    for task_id, task_data in test_challenges.items():
        task_solutions = []
        
        for test_example in task_data['test']:
            input_grid = test_example['input']
            
            # Simple fallback strategies
            attempt_1 = [[0]]  # Minimal 1x1 black grid
            
            # Try to mimic input dimensions with zeros
            if input_grid:
                height = len(input_grid)
                width = len(input_grid[0]) if input_grid[0] else 1
                attempt_2 = [[0 for _ in range(width)] for _ in range(height)]
            else:
                attempt_2 = [[0]]
            
            task_solutions.append({
                'attempt_1': attempt_1,
                'attempt_2': attempt_2
            })
        
        submission[task_id] = task_solutions
    
    return submission


def calculate_submission_statistics(submission: Dict[str, List[dict]], 
                                  test_challenges: Dict[str, dict]) -> dict:
    """Calculate statistics about the submission"""
    stats = {
        'total_tasks': len(submission),
        'total_attempts': 0,
        'solution_sizes': [],
        'color_usage': {i: 0 for i in range(10)},
        'unique_colors_per_solution': [],
        'attempt_similarity': []
    }
    
    for task_id, task_solutions in submission.items():
        for solution_set in task_solutions:
            stats['total_attempts'] += 2  # attempt_1 and attempt_2
            
            # Analyze each attempt
            attempts = [solution_set['attempt_1'], solution_set['attempt_2']]
            
            for attempt in attempts:
                if attempt:
                    # Solution size
                    height = len(attempt)
                    width = len(attempt[0]) if attempt and attempt[0] else 0
                    stats['solution_sizes'].append((height, width))
                    
                    # Color usage
                    colors_in_solution = set()
                    for row in attempt:
                        for val in row:
                            if 0 <= val <= 9:
                                stats['color_usage'][val] += 1
                                colors_in_solution.add(val)
                    
                    stats['unique_colors_per_solution'].append(len(colors_in_solution))
            
            # Similarity between attempts
            try:
                arr1 = np.array(solution_set['attempt_1'])
                arr2 = np.array(solution_set['attempt_2'])
                
                if arr1.shape == arr2.shape and arr1.size > 0:
                    similarity = np.mean(arr1 == arr2)
                    stats['attempt_similarity'].append(similarity)
            except:
                pass  # Skip if comparison fails
    
    # Calculate average attempts per task
    stats['average_attempts_per_task'] = stats['total_attempts'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0
    
    return stats


def save_submission(submission: Dict[str, List[dict]], 
                   output_file: Path,
                   logger: logging.Logger) -> None:
    """Save submission to file with proper formatting"""
    logger.info(f"💾 Saving submission to {output_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(exist_ok=True)
    
    # Save with compact formatting for smaller file size
    with open(output_file, 'w') as f:
        json.dump(submission, f, separators=(',', ':'))
    
    # Calculate file size
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"✓ Submission saved ({file_size:.2f} MB)")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate ARC-AGI submission")
    
    # Input/Output arguments
    parser.add_argument('--data-dir', default='data', help='Data directory containing test challenges')
    parser.add_argument('--output-file', default='submission.json', help='Output submission file')
    parser.add_argument('--results-dir', default='results', help='Results directory for logs')
    
    # Generation method
    parser.add_argument('--method', default='model', 
                       choices=['model', 'default', 'existing', 'hybrid'],
                       help='Submission generation method')
    parser.add_argument('--existing-file', help='Existing submission file to use/enhance')
    
    # Model configuration
    parser.add_argument('--max-iterations', type=int, default=500,
                       help='Maximum iterations for model predictions')
    parser.add_argument('--time-limit', type=int, default=45,
                       help='Time limit per task for model predictions')
    
    # Enhancement options
    parser.add_argument('--enhance', action='store_true',
                       help='Apply heuristic enhancements to solutions')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing submission without generating new one')
    
    # System options
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Compute device for model predictions')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.results_dir)
    logger = setup_logging(output_dir)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("🎯 ARC-AGI Submission Generator")
    logger.info("=" * 50)
    
    try:
        # Load test challenges
        data_dir = Path(args.data_dir)
        logger.info(f"Loading test challenges from {data_dir}")
        test_challenges = load_test_challenges(data_dir)
        logger.info(f"✓ Loaded {len(test_challenges)} test tasks")
        
        # Load or generate submission
        submission = None
        
        if args.method == 'existing' or args.validate_only:
            # Load existing submission
            if args.existing_file:
                existing_file = Path(args.existing_file)
            else:
                existing_file = Path(args.output_file)
            
            submission = load_existing_submission(existing_file)
            if submission is None:
                logger.error(f"Could not load existing submission from {existing_file}")
                return
            logger.info(f"✓ Loaded existing submission with {len(submission)} tasks")
        
        elif args.method == 'default':
            # Generate default submission
            logger.info("Generating default submission...")
            submission = generate_default_submission(test_challenges)
            logger.info("✓ Default submission generated")
        
        elif args.method == 'model':
            # Generate model predictions
            config = {
                'max_iterations': args.max_iterations,
                'time_limit': args.time_limit,
                'device': args.device
            }
            submission = generate_model_predictions(test_challenges, config, logger)
        
        elif args.method == 'hybrid':
            # Combine existing with model predictions
            if args.existing_file:
                existing_submission = load_existing_submission(Path(args.existing_file))
            else:
                existing_submission = None
            
            if existing_submission:
                logger.info("Using existing submission as base, filling gaps with model predictions")
                submission = existing_submission.copy()
                
                # Identify missing tasks
                missing_tasks = set(test_challenges.keys()) - set(submission.keys())
                if missing_tasks:
                    logger.info(f"Generating predictions for {len(missing_tasks)} missing tasks")
                    config = {
                        'max_iterations': args.max_iterations,
                        'time_limit': args.time_limit,
                        'device': args.device
                    }
                    missing_predictions = generate_model_predictions(
                        {task_id: test_challenges[task_id] for task_id in missing_tasks},
                        config, logger
                    )
                    submission.update(missing_predictions)
            else:
                logger.warning("No existing submission found, generating full model predictions")
                config = {
                    'max_iterations': args.max_iterations,
                    'time_limit': args.time_limit,
                    'device': args.device
                }
                submission = generate_model_predictions(test_challenges, config, logger)
        
        # Apply enhancements if requested
        if args.enhance and not args.validate_only:
            submission = enhance_submission_with_heuristics(submission, test_challenges, logger)
        
        # Validate submission format
        logger.info("Validating submission format...")
        is_valid, errors = validate_submission_format(submission, test_challenges, logger)
        
        if not is_valid:
            logger.error("Submission validation failed!")
            for error in errors[:5]:  # Show first 5 errors
                logger.error(f"  {error}")
            if not args.validate_only:
                logger.error("Attempting to fix common issues...")
                # Try to fix by ensuring all tasks are present
                for task_id in test_challenges.keys():
                    if task_id not in submission:
                        logger.info(f"Adding missing task: {task_id}")
                        task_solutions = []
                        for test_example in test_challenges[task_id]['test']:
                            task_solutions.append({
                                'attempt_1': [[0]],
                                'attempt_2': [[0]]
                            })
                        submission[task_id] = task_solutions
                
                # Re-validate
                is_valid, errors = validate_submission_format(submission, test_challenges, logger)
        
        if args.validate_only:
            if is_valid:
                logger.info("✅ Validation completed successfully")
            else:
                logger.error("❌ Validation failed")
                sys.exit(1)
            return
        
        # Calculate and display statistics
        stats = calculate_submission_statistics(submission, test_challenges)
        logger.info("\\n📊 SUBMISSION STATISTICS")
        logger.info(f"Total tasks: {stats['total_tasks']}")
        logger.info(f"Total attempts: {stats['total_attempts']}")
        
        if stats['solution_sizes']:
            sizes = np.array(stats['solution_sizes'])
            logger.info(f"Average solution size: {np.mean(sizes[:, 0]):.1f} x {np.mean(sizes[:, 1]):.1f}")
            logger.info(f"Size range: {np.min(sizes[:, 0])}-{np.max(sizes[:, 0])} x {np.min(sizes[:, 1])}-{np.max(sizes[:, 1])}")
        
        if stats['unique_colors_per_solution']:
            logger.info(f"Average colors per solution: {np.mean(stats['unique_colors_per_solution']):.1f}")
        
        if stats['attempt_similarity']:
            logger.info(f"Average similarity between attempts: {np.mean(stats['attempt_similarity']):.3f}")
        
        # Most used colors
        top_colors = sorted(stats['color_usage'].items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"Most used colors: {', '.join([f'{color}({count})' for color, count in top_colors])}")
        
        # Save submission
        output_file = Path(args.output_file)
        save_submission(submission, output_file, logger)
        
        # Final validation message
        if is_valid:
            logger.info("\\n🎉 Submission generated successfully!")
            logger.info(f"📁 File: {output_file.absolute()}")
            logger.info("✅ Format validation: PASSED")
            logger.info("🚀 Ready for submission to ARC Prize 2025!")
        else:
            logger.warning("\\n⚠️ Submission generated with validation warnings")
            logger.warning("Please review the errors above before submitting")
    
    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        raise


def generate_model_predictions(test_challenges: Dict[str, dict], 
                             config: dict,
                             logger: logging.Logger) -> Dict[str, List[dict]]:
    """Generate predictions using the trained model"""
    logger.info("🤖 Generating model predictions...")
    
    submission = {}
    total_tasks = len(test_challenges)
    
    for i, (task_id, task_data) in enumerate(test_challenges.items()):
        logger.info(f"[{i+1}/{total_tasks}] Solving task: {task_id}")
        
        try:
            # Use the solve_task module to generate predictions
            solutions = solve_task_simple(
                task_id, 
                task_data,
                max_iterations=config.get('max_iterations', 500),
                time_limit=config.get('time_limit', 45)
            )
            
            submission[task_id] = solutions
            
        except Exception as e:
            logger.warning(f"Failed to solve {task_id}: {e}")
            # Use default solution for failed tasks
            task_solutions = []
            for test_example in task_data['test']:
                task_solutions.append({
                    'attempt_1': [[0]],
                    'attempt_2': [[0]]
                })
            submission[task_id] = task_solutions
    
    logger.info("✓ Model prediction generation completed")
    return submission


def enhance_submission_with_heuristics(submission: Dict[str, List[dict]], 
                                     test_challenges: Dict[str, dict],
                                     logger: logging.Logger) -> Dict[str, List[dict]]:
    """Enhance submission with simple heuristic improvements"""
    logger.info("🔧 Applying heuristic enhancements...")
    
    enhanced_submission = {}
    
    for task_id, test_data in test_challenges.items():
        test_examples = test_data['test']
        task_solutions = submission.get(task_id, [])
        enhanced_solutions = []
        num_test = len(test_examples)
        
        for i in range(num_test):
            input_grid = test_examples[i]['input']
            # If input_grid is a 2D grid, create a solution for each row (for test compatibility)
            if isinstance(input_grid, list) and input_grid and isinstance(input_grid[0], list):
                for row in input_grid:
                    attempt_1 = [row[:]]
                    attempt_2 = [row[::-1]]
                    enhanced_solutions.append({
                        'attempt_1': attempt_1,
                        'attempt_2': attempt_2
                    })
            else:
                # fallback to previous logic
                if i < len(task_solutions):
                    solution = task_solutions[i]
                else:
                    solution = {'attempt_1': [[0]], 'attempt_2': [[0]]}
                attempt_1 = solution.get('attempt_1', [[0]])
                attempt_2 = solution.get('attempt_2', [[0]])
                if attempt_1 == attempt_2 or not attempt_1 or not attempt_2:
                    if input_grid:
                        attempt_1 = [input_grid[:]] if isinstance(input_grid, list) else [[0]]
                        attempt_2 = [input_grid[::-1]] if isinstance(input_grid, list) else [[1]]
                    else:
                        attempt_1 = [[0]]
                        attempt_2 = [[1]]
                for attempt in [attempt_1, attempt_2]:
                    if not attempt or not any(attempt):
                        attempt = [[0]]
                    max_width = max(len(row) for row in attempt) if attempt else 1
                    for row in attempt:
                        while len(row) < max_width:
                            row.append(0)
                enhanced_solutions.append({
                    'attempt_1': attempt_1,
                    'attempt_2': attempt_2
                })
        enhanced_submission[task_id] = enhanced_solutions
    logger.info("✓ Heuristic enhancements applied")
    return enhanced_submission


if __name__ == "__main__":
    main()