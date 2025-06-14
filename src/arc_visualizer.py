import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from pathlib import Path


# Enhanced ARC color palette with better visualization
ARC_COLORS = np.array([
    [0, 0, 0],          # 0: black
    [30, 147, 255],     # 1: blue
    [249, 60, 49],      # 2: red
    [79, 204, 48],      # 3: green
    [255, 220, 0],      # 4: yellow
    [153, 153, 153],    # 5: gray
    [229, 58, 163],     # 6: magenta
    [255, 133, 27],     # 7: orange
    [135, 216, 241],    # 8: light blue
    [146, 18, 49],      # 9: brown
    [255, 255, 255],    # 10: white (for background/padding)
]) / 255.0

# Create custom colormap
ARC_COLORMAP = ListedColormap(ARC_COLORS)


class EnhancedVisualizer:
    """Enhanced visualization with advanced features."""
    
    def __init__(self, output_dir: str = "outputs/plots", dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
    
    def convert_color(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to RGB colors with enhanced handling."""
        if grid.ndim == 2:
            # Add channel dimension
            grid = grid[..., np.newaxis]
        
        # Ensure values are in valid range
        grid = np.clip(grid.astype(int), 0, len(ARC_COLORS) - 1)
        
        # Convert to RGB
        rgb = np.zeros(grid.shape[:2] + (3,))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color_idx = grid[i, j, 0] if grid.ndim == 3 else grid[i, j]
                rgb[i, j] = ARC_COLORS[color_idx]
        
        return (rgb * 255).astype(np.uint8)
    
    def plot_grid_with_borders(self, ax, grid: np.ndarray, title: str = "", 
                              show_coordinates: bool = False, highlight_differences: Optional[np.ndarray] = None):
        """Plot grid with enhanced borders and optional features."""
        
        # Convert to colors
        rgb_grid = self.convert_color(grid)
        
        # Plot the grid
        ax.imshow(rgb_grid, interpolation='nearest')
        
        # Add grid lines
        for i in range(grid.shape[0] + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.7)
        for j in range(grid.shape[1] + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.7)
        
        # Highlight differences if provided
        if highlight_differences is not None:
            diff_y, diff_x = np.where(highlight_differences)
            for y, x in zip(diff_y, diff_x):
                rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, 
                                       linewidth=2, edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
        
        # Show coordinates if requested
        if show_coordinates:
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    value = grid[i, j] if grid.ndim == 2 else grid[i, j, 0]
                    ax.text(j, i, str(value), ha='center', va='center', 
                           fontsize=8, color='white', weight='bold')
        
        # Set title and clean up axes
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    def plot_uncertainty_heatmap(self, ax, uncertainty: np.ndarray, title: str = "Uncertainty"):
        """Plot uncertainty as a heatmap."""
        im = ax.imshow(uncertainty, cmap='Reds', interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        return im


def plot_enhanced_problem(logger, save_individual: bool = True, show_stats: bool = True):
    """Enhanced problem visualization with statistics and analysis."""
    
    visualizer = EnhancedVisualizer()
    task = logger.task
    
    # Calculate layout
    n_train = task.n_train
    n_test = task.n_test
    n_examples = task.n_examples
    
    # Create figure with subplots
    fig_width = max(12, n_examples * 3)
    fig_height = max(8, 6 + (2 if show_stats else 0))
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid layout
    if show_stats:
        gs = fig.add_gridspec(3, n_examples, height_ratios=[0.3, 1, 1], hspace=0.3, wspace=0.2)
        stats_ax = fig.add_subplot(gs[0, :])
    else:
        gs = fig.add_gridspec(2, n_examples, hspace=0.3, wspace=0.2)
    
    # Plot statistics if requested
    if show_stats and hasattr(task, 'stats') and task.stats:
        plot_task_statistics(stats_ax, task.stats)
    
    # Plot examples
    for example_num in range(n_examples):
        # Determine split and local index
        if example_num < n_train:
            split = 'train'
            local_idx = example_num
        else:
            split = 'test'
            local_idx = example_num - n_train
        
        # Get grids
        example_data = task.unprocessed_problem[split][local_idx]
        input_grid = np.array(example_data['input'])
        
        # Plot input
        input_ax = fig.add_subplot(gs[-2, example_num])
        visualizer.plot_grid_with_borders(
            input_ax, input_grid, 
            f"{split.title()} {local_idx + 1}\nInput ({input_grid.shape[0]}×{input_grid.shape[1]})"
        )
        
        # Plot output
        output_ax = fig.add_subplot(gs[-1, example_num])
        
        if 'output' in example_data:
            output_grid = np.array(example_data['output'])
            
            # Check for differences if same size
            differences = None
            if input_grid.shape == output_grid.shape:
                differences = input_grid != output_grid
            
            visualizer.plot_grid_with_borders(
                output_ax, output_grid,
                f"Output ({output_grid.shape[0]}×{output_grid.shape[1]})",
                highlight_differences=differences
            )
        else:
            # Test case - show question mark
            output_ax.text(0.5, 0.5, '?', fontsize=48, ha='center', va='center',
                          transform=output_ax.transAxes, fontweight='bold')
            output_ax.set_title("Unknown Output")
            output_ax.set_xticks([])
            output_ax.set_yticks([])
    
    # Add main title
    main_title = f"ARC Task: {task.task_name}"
    if hasattr(task, 'stats') and task.stats:
        main_title += f" (Difficulty: {task.stats.difficulty_score:.2f})"
    
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # Save plot
    output_file = visualizer.output_dir / f"{task.task_name}_problem.png"
    plt.savefig(output_file, dpi=visualizer.dpi, bbox_inches='tight', facecolor='white')
    
    if save_individual:
        plt.savefig(visualizer.output_dir / f"{task.task_name}_problem.pdf", 
                   bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    print(f"Enhanced problem plot saved: {output_file}")


def plot_task_statistics(ax, stats):
    """Plot task statistics in a compact format."""
    
    # Create statistics text
    stats_text = [
        f"Examples: {stats.n_train_examples} train, {stats.n_test_examples} test",
        f"Grid Size: {stats.max_grid_size}×{stats.max_grid_size} max",
        f"Colors: {stats.color_diversity}",
        f"Complexity: {stats.pattern_complexity:.2f}",
        f"Type: {stats.transformation_type}",
        f"Difficulty: {stats.difficulty_score:.2f}"
    ]
    
    # Plot as text boxes
    ax.text(0.02, 0.5, " | ".join(stats_text), transform=ax.transAxes,
           fontsize=10, va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def plot_enhanced_solution(logger, save_individual: bool = True, show_uncertainty: bool = True,
                          show_ensemble: bool = True):
    """Enhanced solution visualization with uncertainty and ensemble analysis."""
    
    visualizer = EnhancedVisualizer()
    task = logger.task
    
    n_test = task.n_test
    
    # Get solutions
    solutions_data = []
    
    # Current solution
    if hasattr(logger, 'current_logits') and logger.current_logits is not None:
        current_probs = torch.softmax(logger.current_logits, dim=1).cpu().numpy()
        solutions_data.append(('Current', current_probs, True))
    
    # EMA solutions
    if hasattr(logger, 'ema_logits_fast') and logger.ema_logits_fast is not None:
        ema_fast_probs = torch.softmax(logger.ema_logits_fast, dim=1).cpu().numpy()
        solutions_data.append(('EMA Fast', ema_fast_probs, True))
    
    if hasattr(logger, 'ema_logits_slow') and logger.ema_logits_slow is not None:
        ema_slow_probs = torch.softmax(logger.ema_logits_slow, dim=1).cpu().numpy()
        solutions_data.append(('EMA Slow', ema_slow_probs, True))
    
    # Discrete solutions
    discrete_solutions = [
        ('Most Frequent', logger.solution_most_frequent, False),
        ('Second Most Frequent', logger.solution_second_most_frequent, False),
    ]
    
    if hasattr(logger, 'solution_highest_confidence'):
        discrete_solutions.append(('Highest Confidence', logger.solution_highest_confidence, False))
    
    if hasattr(logger, 'solution_most_consistent'):
        discrete_solutions.append(('Most Consistent', logger.solution_most_consistent, False))
    
    solutions_data.extend(discrete_solutions)
    
    # Create figure
    n_solutions = len(solutions_data)
    n_cols = min(n_solutions, 4)
    n_rows = (n_solutions + n_cols - 1) // n_cols
    
    if show_uncertainty:
        n_rows *= 2  # Double rows for uncertainty
    
    fig_width = max(12, n_cols * 3)
    fig_height = max(8, n_rows * 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot solutions
    for sol_idx, (sol_name, sol_data, is_probabilistic) in enumerate(solutions_data):
        row = (sol_idx // n_cols) * (2 if show_uncertainty else 1)
        col = sol_idx % n_cols
        
        if row >= n_rows or col >= n_cols:
            continue
        
        # Plot main solution
        ax_main = axes[row, col]
        
        for test_idx in range(min(n_test, 1)):  # Show first test example
            if is_probabilistic:
                # Convert probabilities to discrete prediction
                pred_colors = np.argmax(sol_data[test_idx], axis=0)
                
                # Get uncertainty
                uncertainty = None
                if show_uncertainty:
                    probs = sol_data[test_idx]
                    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)
                    uncertainty = entropy / np.log(probs.shape[0])  # Normalized entropy
                
                # Crop to reasonable size
                pred_grid = crop_solution_grid(pred_colors, task, test_idx)
                
                visualizer.plot_grid_with_borders(ax_main, pred_grid, f"{sol_name}\n{pred_grid.shape}")
                
                # Plot uncertainty if available
                if show_uncertainty and uncertainty is not None and row + 1 < n_rows:
                    ax_uncertainty = axes[row + 1, col]
                    uncertainty_cropped = crop_solution_grid(uncertainty, task, test_idx)
                    visualizer.plot_uncertainty_heatmap(ax_uncertainty, uncertainty_cropped, 
                                                      f"{sol_name} Uncertainty")
            
            else:
                # Discrete solution
                if sol_data and test_idx < len(sol_data):
                    discrete_grid = np.array(sol_data[test_idx])
                    visualizer.plot_grid_with_borders(ax_main, discrete_grid, 
                                                    f"{sol_name}\n{discrete_grid.shape}")
                else:
                    ax_main.text(0.5, 0.5, 'No Solution', ha='center', va='center',
                               transform=ax_main.transAxes)
                    ax_main.set_title(sol_name)
    
    # Hide unused subplots
    for i in range(len(solutions_data), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if row < n_rows and col < n_cols:
            axes[row, col].axis('off')
    
    # Add main title
    fig.suptitle(f"Solutions: {task.task_name}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = visualizer.output_dir / f"{task.task_name}_solutions.png"
    plt.savefig(output_file, dpi=visualizer.dpi, bbox_inches='tight', facecolor='white')
    
    if save_individual:
        plt.savefig(visualizer.output_dir / f"{task.task_name}_solutions.pdf", 
                   bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    print(f"Enhanced solution plot saved: {output_file}")


def crop_solution_grid(grid: np.ndarray, task, test_idx: int, max_size: int = 20) -> np.ndarray:
    """Crop solution grid to reasonable size for visualization."""
    
    # Get expected output shape if available
    expected_shape = None
    if hasattr(task, 'shapes') and test_idx + task.n_train < len(task.shapes):
        expected_shape = task.shapes[test_idx + task.n_train][1]
    
    if expected_shape:
        h, w = expected_shape
        return grid[:h, :w]
    
    # Automatic cropping based on content
    if grid.ndim == 2 and grid.size > max_size * max_size:
        # Find bounding box of non-zero content
        nonzero_rows = np.any(grid != 0, axis=1)
        nonzero_cols = np.any(grid != 0, axis=0)
        
        if np.any(nonzero_rows) and np.any(nonzero_cols):
            first_row, last_row = np.where(nonzero_rows)[0][[0, -1]]
            first_col, last_col = np.where(nonzero_cols)[0][[0, -1]]
            
            # Add small padding
            first_row = max(0, first_row - 1)
            last_row = min(grid.shape[0] - 1, last_row + 1)
            first_col = max(0, first_col - 1)
            last_col = min(grid.shape[1] - 1, last_col + 1)
            
            return grid[first_row:last_row + 1, first_col:last_col + 1]
    
    # Fallback: crop to max_size
    return grid[:max_size, :max_size]


def plot_training_metrics(logger, save_path: Optional[str] = None):
    """Plot comprehensive training metrics."""
    
    visualizer = EnhancedVisualizer()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot 1: Loss curves
    if logger.metrics_history['loss_curve']:
        axes[0].plot(logger.metrics_history['loss_curve'], label='Total Loss', linewidth=2)
        axes[0].plot(logger.metrics_history['reconstruction_error_curve'], 
                    label='Reconstruction', alpha=0.7)
        axes[0].plot(logger.metrics_history['total_KL_curve'], 
                    label='KL Divergence', alpha=0.7)
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
    
    # Plot 2: KL components
    if logger.metrics_history['KL_curves']:
        for name, curve in logger.metrics_history['KL_curves'].items():
            axes[1].plot(curve, label=name, alpha=0.7)
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('KL Divergence')
        axes[1].set_title('KL Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    if logger.metrics_history['learning_rate_curve']:
        axes[2].plot(logger.metrics_history['learning_rate_curve'], linewidth=2)
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    # Plot 4: Gradient norms
    if logger.metrics_history['gradient_norm_curve']:
        axes[3].plot(logger.metrics_history['gradient_norm_curve'], linewidth=2)
        axes[3].set_xlabel('Training Step')
        axes[3].set_ylabel('Gradient Norm')
        axes[3].set_title('Gradient Norms')
        axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Solution diversity
    if hasattr(logger, 'solution_tracker') and logger.solution_tracker.solutions:
        steps = range(len(logger.solution_contributions_log))
        unique_solutions = [len(set(contrib[0] for contrib in step_contribs)) 
                           for step_contribs in logger.solution_contributions_log]
        axes[4].plot(steps, unique_solutions, linewidth=2)
        axes[4].set_xlabel('Training Step')
        axes[4].set_ylabel('Unique Solutions')
        axes[4].set_title('Solution Diversity')
        axes[4].grid(True, alpha=0.3)
    
    # Plot 6: Solution quality evolution
    if hasattr(logger, 'solution_tracker') and logger.solution_tracker.quality_scores:
        qualities = [q.overall_score for q in logger.solution_tracker.quality_scores]
        confidences = [q.confidence for q in logger.solution_tracker.quality_scores]
        
        axes[5].scatter(qualities, confidences, alpha=0.6, s=20)
        axes[5].set_xlabel('Overall Quality')
        axes[5].set_ylabel('Confidence')
        axes[5].set_title('Quality vs Confidence')
        axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = visualizer.output_dir / f"{logger.task.task_name}_training_metrics.png"
    
    plt.savefig(save_path, dpi=visualizer.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Training metrics plot saved: {save_path}")


def create_ensemble_comparison(loggers: List, save_path: Optional[str] = None):
    """Create ensemble comparison visualization."""
    
    visualizer = EnhancedVisualizer()
    
    n_tasks = len(loggers)
    n_cols = min(4, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_tasks > 1 else [axes]
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, logger in enumerate(loggers):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col] if n_tasks > 1 else axes
        else:
            ax = axes[row, col]
        
        # Plot solution quality evolution
        if hasattr(logger, 'solution_tracker') and logger.solution_tracker.quality_scores:
            timestamps = logger.solution_tracker.timestamps
            qualities = [q.overall_score for q in logger.solution_tracker.quality_scores]
            
            ax.scatter(timestamps, qualities, alpha=0.6, s=20)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Solution Quality')
            ax.set_title(f'Task: {logger.task.task_name}')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(timestamps) > 1:
                z = np.polyfit(timestamps, qualities, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(timestamps), "r--", alpha=0.8, linewidth=2)
    
    # Hide unused subplots
    for i in range(n_tasks, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].axis('off') if n_tasks > 1 else None
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = visualizer.output_dir / "ensemble_comparison.png"
    
    plt.savefig(save_path, dpi=visualizer.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Ensemble comparison saved: {save_path}")


# Backward compatibility functions
def plot_problem(logger):
    """Backward compatible problem plotting."""
    plot_enhanced_problem(logger, save_individual=False, show_stats=False)


def plot_solution(logger, fname: Optional[str] = None):
    """Backward compatible solution plotting."""
    plot_enhanced_solution(logger, save_individual=(fname is not None), 
                          show_uncertainty=False, show_ensemble=False)