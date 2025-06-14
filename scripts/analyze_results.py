import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

def load_results(results_file: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance metrics from results."""
    
    successful_tasks = [r for r in results.values() if 'solutions' in r]
    failed_tasks = [r for r in results.values() if 'solutions' not in r]
    
    analysis = {
        'total_tasks': len(results),
        'successful_tasks': len(successful_tasks),
        'failed_tasks': len(failed_tasks),
        'success_rate': len(successful_tasks) / len(results) * 100
    }
    
    if successful_tasks:
        memory_usage = [r.get('memory_used', 0) for r in successful_tasks]
        iterations = [r.get('iterations_completed', 0) for r in successful_tasks]
        
        analysis.update({
            'avg_memory_mb': np.mean(memory_usage) / 1e6,
            'max_memory_mb': np.max(memory_usage) / 1e6,
            'avg_iterations': np.mean(iterations),
            'max_iterations': np.max(iterations)
        })
    
    return analysis

def create_visualizations(results: Dict[str, Any], output_dir: str = "outputs") -> None:
    """Create performance visualizations."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    successful_tasks = [r for r in results.values() if 'solutions' in r]
    
    if not successful_tasks:
        print("No successful tasks to visualize")
        return
    
    # Memory usage distribution
    memory_usage = [r.get('memory_used', 0) / 1e6 for r in successful_tasks]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.hist(memory_usage, bins=20, alpha=0.7)
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Number of Tasks')
    plt.title('Memory Usage Distribution')
    
    # Iterations distribution
    iterations = [r.get('iterations_completed', 0) for r in successful_tasks]
    
    plt.subplot(2, 2, 2)
    plt.hist(iterations, bins=20, alpha=0.7)
    plt.xlabel('Iterations Completed')
    plt.ylabel('Number of Tasks')
    plt.title('Training Iterations Distribution')
    
    # Success rate pie chart
    analysis = analyze_performance(results)
    
    plt.subplot(2, 2, 3)
    labels = ['Successful', 'Failed']
    sizes = [analysis['successful_tasks'], analysis['failed_tasks']]
    colors = ['lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Task Success Rate')
    
    # Memory vs Iterations scatter
    plt.subplot(2, 2, 4)
    plt.scatter(iterations, memory_usage, alpha=0.6)
    plt.xlabel('Iterations Completed')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory vs Iterations')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / "performance_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {plot_file}")

def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print formatted analysis report."""
    
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*50)
    
    print(f"Total Tasks: {analysis['total_tasks']}")
    print(f"Successful Tasks: {analysis['successful_tasks']}")
    print(f"Failed Tasks: {analysis['failed_tasks']}")
    print(f"Success Rate: {analysis['success_rate']:.1f}%")
    
    if 'avg_memory_mb' in analysis:
        print(f"\nMemory Usage:")
        print(f"  Average: {analysis['avg_memory_mb']:.1f} MB")
        print(f"  Maximum: {analysis['max_memory_mb']:.1f} MB")
        
        print(f"\nTraining Iterations:")
        print(f"  Average: {analysis['avg_iterations']:.1f}")
        print(f"  Maximum: {analysis['max_iterations']}")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Analyze ARC results")
    parser.add_argument("results_file", help="Results JSON file to analyze")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Load and analyze results
    results = load_results(args.results_file)
    analysis = analyze_performance(results)
    
    # Print report
    print_analysis_report(analysis)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(results, args.output_dir)

if __name__ == "__main__":
    import argparse
    main()