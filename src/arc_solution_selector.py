"""
Enhanced solution selection with advanced tracking, uncertainty estimation, and ensemble methods.
Optimized for competition performance with quality metrics and adaptive selection.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings


@dataclass
class SolutionQuality:
    """Quality metrics for a solution."""
    confidence: float
    uncertainty: float
    consistency: float
    novelty: float
    overall_score: float


class UncertaintyEstimator:
    """Advanced uncertainty estimation for solution quality."""
    
    def __init__(self, calibration_factor: float = 1.0):
        self.calibration_factor = calibration_factor
        self.history = deque(maxlen=1000)
    
    def estimate_epistemic_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Estimate epistemic (model) uncertainty using entropy."""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        # Normalize by maximum possible entropy
        max_entropy = torch.log(torch.tensor(logits.shape[1], dtype=logits.dtype, device=logits.device))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def estimate_aleatoric_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Estimate aleatoric (data) uncertainty using predictive variance."""
        # Sample multiple predictions with dropout-like noise
        n_samples = 10
        noise_std = 0.1
        
        predictions = []
        for _ in range(n_samples):
            noisy_logits = logits + torch.randn_like(logits) * noise_std
            probs = F.softmax(noisy_logits, dim=1)
            predictions.append(probs)
        
        # Calculate variance across samples
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        variance = torch.mean(torch.var(predictions, dim=0), dim=1)
        
        return variance
    
    def estimate_total_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Estimate total uncertainty combining epistemic and aleatoric."""
        epistemic = self.estimate_epistemic_uncertainty(logits)
        aleatoric = self.estimate_aleatoric_uncertainty(logits)
        
        # Weighted combination with calibration
        total_uncertainty = (0.6 * epistemic + 0.4 * aleatoric) * self.calibration_factor
        
        return total_uncertainty
    
    def calibrate_uncertainty(self, uncertainty: torch.Tensor, accuracy: torch.Tensor):
        """Calibrate uncertainty estimates based on observed accuracy."""
        if len(self.history) > 10:
            # Compute expected calibration error
            hist_uncertainty = torch.tensor([h[0] for h in self.history])
            hist_accuracy = torch.tensor([h[1] for h in self.history])
            
            # Simple linear calibration
            if torch.std(hist_uncertainty) > 1e-6:
                correlation = torch.corrcoef(torch.stack([hist_uncertainty, hist_accuracy]))[0, 1]
                self.calibration_factor *= (1.0 + 0.1 * correlation.item())
                self.calibration_factor = np.clip(self.calibration_factor, 0.5, 2.0)
        
        # Store for future calibration
        self.history.append((uncertainty.mean().item(), accuracy.mean().item()))


class SolutionTracker:
    """Advanced solution tracking with quality metrics."""
    
    def __init__(self, max_solutions: int = 100):
        self.max_solutions = max_solutions
        self.solutions = []
        self.quality_scores = []
        self.timestamps = []
        self.uncertainties = []
        
    def add_solution(self, solution: Tuple, quality: SolutionQuality, timestamp: float):
        """Add a solution with quality metrics."""
        self.solutions.append(solution)
        self.quality_scores.append(quality)
        self.timestamps.append(timestamp)
        
        # Keep only best solutions
        if len(self.solutions) > self.max_solutions:
            # Sort by overall score and keep top solutions
            sorted_indices = sorted(
                range(len(self.quality_scores)), 
                key=lambda i: self.quality_scores[i].overall_score, 
                reverse=True
            )
            
            keep_indices = sorted_indices[:self.max_solutions]
            
            self.solutions = [self.solutions[i] for i in keep_indices]
            self.quality_scores = [self.quality_scores[i] for i in keep_indices]
            self.timestamps = [self.timestamps[i] for i in keep_indices]
    
    def get_best_solutions(self, n: int = 2) -> List[Tuple]:
        """Get n best solutions based on quality scores."""
        if not self.solutions:
            return [tuple(((0, 0), (0, 0)) for _ in range(n))]
        
        # Sort by overall score
        sorted_indices = sorted(
            range(len(self.quality_scores)), 
            key=lambda i: self.quality_scores[i].overall_score, 
            reverse=True
        )
        
        best_solutions = []
        for i in range(min(n, len(sorted_indices))):
            idx = sorted_indices[i]
            best_solutions.append(self.solutions[idx])
        
        # Fill remaining slots with best solution if needed
        while len(best_solutions) < n:
            best_solutions.append(self.solutions[sorted_indices[0]])
        
        return best_solutions
    
    def get_diversity_weighted_solutions(self, n: int = 2) -> List[Tuple]:
        """Get diverse solutions using quality and diversity metrics."""
        if not self.solutions:
            return [tuple(((0, 0), (0, 0)) for _ in range(n))]
        
        selected = []
        remaining = list(range(len(self.solutions)))
        
        # Select first solution (highest quality)
        best_idx = max(remaining, key=lambda i: self.quality_scores[i].overall_score)
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Select remaining solutions balancing quality and diversity
        for _ in range(min(n - 1, len(remaining))):
            diversity_scores = []
            
            for idx in remaining:
                # Calculate diversity from already selected solutions
                diversity = sum(
                    self._solution_diversity(self.solutions[idx], self.solutions[sel_idx])
                    for sel_idx in selected
                ) / len(selected)
                
                # Combined score: quality + diversity
                combined_score = (0.7 * self.quality_scores[idx].overall_score + 
                                0.3 * diversity)
                diversity_scores.append(combined_score)
            
            # Select best combined score
            best_remaining_idx = max(
                range(len(remaining)), 
                key=lambda i: diversity_scores[i]
            )
            selected.append(remaining[best_remaining_idx])
            remaining.remove(remaining[best_remaining_idx])
        
        return [self.solutions[i] for i in selected]
    
    def _solution_diversity(self, sol1: Tuple, sol2: Tuple) -> float:
        """Calculate diversity between two solutions."""
        try:
            # Simple diversity based on grid differences
            diversity = 0.0
            count = 0
            
            for grid1, grid2 in zip(sol1, sol2):
                arr1 = np.array(grid1)
                arr2 = np.array(grid2)
                
                if arr1.shape == arr2.shape:
                    diff_ratio = np.mean(arr1 != arr2)
                    diversity += diff_ratio
                    count += 1
                else:
                    diversity += 1.0  # Maximum diversity for different shapes
                    count += 1
            
            return diversity / max(count, 1)
            
        except Exception:
            return 0.5  # Default diversity


class EnhancedLogger:
    """Enhanced logging with advanced metrics and solution selection."""
    
    def __init__(self, task, config: Optional[Dict[str, Any]] = None):
        self.task = task
        self.config = config or {}
        
        # Core tracking
        self.solution_tracker = SolutionTracker()
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Training metrics
        self.metrics_history = {
            'KL_curves': {},
            'total_KL_curve': [],
            'reconstruction_error_curve': [],
            'loss_curve': [],
            'uncertainty_curve': [],
            'confidence_curve': [],
            'learning_rate_curve': [],
            'gradient_norm_curve': []
        }
        
        # Current state
        self.current_step = 0
        self.current_logits = None
        self.current_x_mask = None
        self.current_y_mask = None
        
        # EMA tracking with multiple decay rates
        self.ema_decay_fast = 0.9
        self.ema_decay_slow = 0.99
        
        self.ema_logits_fast = None
        self.ema_logits_slow = None
        self.ema_x_mask_fast = None
        self.ema_x_mask_slow = None
        self.ema_y_mask_fast = None
        self.ema_y_mask_slow = None
        
        # Solution management
        self.solution_hashes_count = {}
        self.solution_contributions_log = []
        self.solution_picks_history = []
        
        # Best solutions with enhanced tracking
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None
        self.solution_highest_confidence = None
        self.solution_most_consistent = None
        
        # Initialize default solutions
        self._initialize_default_solutions()
    
    def _initialize_default_solutions(self):
        """Initialize default solutions."""
        default_solution = tuple(((0, 0), (0, 0)) for _ in range(self.task.n_test))
        
        self.solution_most_frequent = default_solution
        self.solution_second_most_frequent = default_solution
        self.solution_highest_confidence = default_solution
        self.solution_most_consistent = default_solution
    
    def log(self, train_step: int, logits: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor,
            KL_amounts: List[torch.Tensor], KL_names: List[str], total_KL: torch.Tensor,
            reconstruction_error: torch.Tensor, loss: torch.Tensor, **kwargs):
        """Enhanced logging with advanced metrics."""
        
        self.current_step = train_step
        
        # Initialize KL curves
        if train_step == 0:
            self.metrics_history['KL_curves'] = {name: [] for name in KL_names}
        
        # Log KL components
        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            kl_value = float(KL_amount.detach().sum().cpu().numpy())
            self.metrics_history['KL_curves'][KL_name].append(kl_value)
        
        # Log main metrics
        self.metrics_history['total_KL_curve'].append(float(total_KL.detach().cpu().numpy()))
        self.metrics_history['reconstruction_error_curve'].append(float(reconstruction_error.detach().cpu().numpy()))
        self.metrics_history['loss_curve'].append(float(loss.detach().cpu().numpy()))
        
        # Log additional metrics
        if 'learning_rate' in kwargs:
            self.metrics_history['learning_rate_curve'].append(kwargs['learning_rate'])
        if 'gradient_norm' in kwargs:
            self.metrics_history['gradient_norm_curve'].append(kwargs['gradient_norm'])
        
        # Enhanced solution tracking
        self._track_enhanced_solution(train_step, logits.detach(), x_mask.detach(), y_mask.detach())
    
    def _track_enhanced_solution(self, train_step: int, logits: torch.Tensor, 
                                x_mask: torch.Tensor, y_mask: torch.Tensor):
        """Enhanced solution tracking with quality metrics."""
        
        # Extract test predictions
        test_logits = logits[self.task.n_train:, :, :, :, 1]  # test outputs only
        test_x_mask = x_mask[self.task.n_train:, :, 1]
        test_y_mask = y_mask[self.task.n_train:, :, 1]
        
        # Update current state
        self.current_logits = test_logits
        self.current_x_mask = test_x_mask
        self.current_y_mask = test_y_mask
        
        # Update EMA estimates
        self._update_ema_estimates(test_logits, test_x_mask, test_y_mask)
        
        # Generate candidate solutions with quality assessment
        candidates = [
            ('current', test_logits, test_x_mask, test_y_mask),
            ('ema_fast', self.ema_logits_fast, self.ema_x_mask_fast, self.ema_y_mask_fast),
            ('ema_slow', self.ema_logits_slow, self.ema_x_mask_slow, self.ema_y_mask_slow),
        ]
        
        solution_contributions = []
        
        for name, pred_logits, pred_x_mask, pred_y_mask in candidates:
            if pred_logits is None:
                continue
            
            # Process solution
            solution, quality = self._process_solution_with_quality(
                pred_logits, pred_x_mask, pred_y_mask, train_step
            )
            
            # Add to tracker
            self.solution_tracker.add_solution(solution, quality, train_step)
            
            # Track contributions
            hashed_solution = hash(solution)
            score = quality.overall_score
            
            # Adaptive scoring based on training progress
            if train_step < 150:
                score *= 0.8  # Reduce confidence early in training
            
            if name == 'ema_slow':
                score *= 1.1  # Boost stable estimates
            
            solution_contributions.append((hashed_solution, score))
            
            # Update hash counts
            self.solution_hashes_count[hashed_solution] = float(
                np.logaddexp(self.solution_hashes_count.get(hashed_solution, -np.inf), score)
            )
        
        # Update best solutions
        self._update_best_solutions()
        
        # Log solution contributions
        self.solution_contributions_log.append(solution_contributions)
        self.solution_picks_history.append([
            hash(sol) for sol in [
                self.solution_most_frequent,
                self.solution_second_most_frequent,
                self.solution_highest_confidence,
                self.solution_most_consistent
            ]
        ])
    
    def _update_ema_estimates(self, logits: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor):
        """Update EMA estimates with multiple decay rates."""
        if self.ema_logits_fast is None:
            # Initialize EMA
            self.ema_logits_fast = logits.clone()
            self.ema_logits_slow = logits.clone()
            self.ema_x_mask_fast = x_mask.clone()
            self.ema_x_mask_slow = x_mask.clone()
            self.ema_y_mask_fast = y_mask.clone()
            self.ema_y_mask_slow = y_mask.clone()
        else:
            # Update fast EMA
            self.ema_logits_fast = (self.ema_decay_fast * self.ema_logits_fast + 
                                   (1 - self.ema_decay_fast) * logits)
            self.ema_x_mask_fast = (self.ema_decay_fast * self.ema_x_mask_fast + 
                                   (1 - self.ema_decay_fast) * x_mask)
            self.ema_y_mask_fast = (self.ema_decay_fast * self.ema_y_mask_fast + 
                                   (1 - self.ema_decay_fast) * y_mask)
            
            # Update slow EMA
            self.ema_logits_slow = (self.ema_decay_slow * self.ema_logits_slow + 
                                   (1 - self.ema_decay_slow) * logits)
            self.ema_x_mask_slow = (self.ema_decay_slow * self.ema_x_mask_slow + 
                                   (1 - self.ema_decay_slow) * x_mask)
            self.ema_y_mask_slow = (self.ema_decay_slow * self.ema_y_mask_slow + 
                                   (1 - self.ema_decay_slow) * y_mask)
    
    def _process_solution_with_quality(self, logits: torch.Tensor, x_mask: torch.Tensor, 
                                     y_mask: torch.Tensor, train_step: int) -> Tuple[Tuple, SolutionQuality]:
        """Process solution and compute quality metrics."""
        
        # Extract solution
        solution, raw_uncertainty = self._postprocess_solution(logits, x_mask, y_mask)
        
        # Estimate uncertainties
        epistemic_uncertainty = self.uncertainty_estimator.estimate_epistemic_uncertainty(
            logits.view(-1, logits.shape[1])
        ).mean().item()
        
        total_uncertainty = self.uncertainty_estimator.estimate_total_uncertainty(
            logits.view(-1, logits.shape[1])
        ).mean().item()
        
        # Calculate confidence
        max_probs = F.softmax(logits, dim=1).max(dim=1)[0]
        confidence = max_probs.mean().item()
        
        # Calculate consistency (if we have history)
        consistency = self._calculate_solution_consistency(solution)
        
        # Calculate novelty
        novelty = self._calculate_solution_novelty(solution)
        
        # Overall quality score
        overall_score = self._calculate_overall_quality(
            confidence, total_uncertainty, consistency, novelty, train_step
        )
        
        quality = SolutionQuality(
            confidence=confidence,
            uncertainty=total_uncertainty,
            consistency=consistency,
            novelty=novelty,
            overall_score=overall_score
        )
        
        return solution, quality
    
    def _calculate_solution_consistency(self, solution: Tuple) -> float:
        """Calculate consistency with previous solutions."""
        if len(self.solution_tracker.solutions) < 2:
            return 0.5  # Default consistency
        
        # Compare with recent solutions
        recent_solutions = self.solution_tracker.solutions[-10:]
        consistency_scores = []
        
        for prev_solution in recent_solutions:
            similarity = 1.0 - self.solution_tracker._solution_diversity(solution, prev_solution)
            consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _calculate_solution_novelty(self, solution: Tuple) -> float:
        """Calculate novelty of the solution."""
        hashed_solution = hash(solution)
        
        # Novelty is inversely related to frequency
        frequency = self.solution_hashes_count.get(hashed_solution, 0)
        novelty = 1.0 / (1.0 + frequency)
        
        return novelty
    
    def _calculate_overall_quality(self, confidence: float, uncertainty: float, 
                                 consistency: float, novelty: float, train_step: int) -> float:
        """Calculate overall quality score."""
        
        # Adaptive weights based on training progress
        progress = min(train_step / 1000.0, 1.0)
        
        # Early training: favor confidence and low uncertainty
        # Late training: favor consistency
        confidence_weight = 0.4 + 0.1 * (1 - progress)
        uncertainty_weight = 0.3 + 0.1 * (1 - progress)
        consistency_weight = 0.2 + 0.1 * progress
        novelty_weight = 0.1
        
        # Combine metrics (uncertainty is inverted)
        overall_score = (
            confidence_weight * confidence +
            uncertainty_weight * (1.0 - uncertainty) +
            consistency_weight * consistency +
            novelty_weight * novelty
        )
        
        return np.clip(overall_score, 0.0, 1.0)
    
    def _update_best_solutions(self):
        """Update best solutions based on different criteria."""
        
        if not self.solution_tracker.solutions:
            return
        
        # Most frequent (highest hash count)
        most_frequent_hash = max(self.solution_hashes_count.items(), key=lambda x: x[1])[0]
        for solution in self.solution_tracker.solutions:
            if hash(solution) == most_frequent_hash:
                self.solution_most_frequent = solution
                break
        
        # Second most frequent
        sorted_hashes = sorted(self.solution_hashes_count.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_hashes) > 1:
            second_frequent_hash = sorted_hashes[1][0]
            for solution in self.solution_tracker.solutions:
                if hash(solution) == second_frequent_hash:
                    self.solution_second_most_frequent = solution
                    break
        
        # Highest confidence
        if self.solution_tracker.quality_scores:
            best_confidence_idx = max(
                range(len(self.solution_tracker.quality_scores)),
                key=lambda i: self.solution_tracker.quality_scores[i].confidence
            )
            self.solution_highest_confidence = self.solution_tracker.solutions[best_confidence_idx]
        
        # Most consistent
        if self.solution_tracker.quality_scores:
            best_consistency_idx = max(
                range(len(self.solution_tracker.quality_scores)),
                key=lambda i: self.solution_tracker.quality_scores[i].consistency
            )
            self.solution_most_consistent = self.solution_tracker.solutions[best_consistency_idx]
    
    def _postprocess_solution(self, prediction: torch.Tensor, x_mask: torch.Tensor, 
                            y_mask: torch.Tensor) -> Tuple[Tuple, float]:
        """Enhanced solution postprocessing with uncertainty."""
        
        # Convert to colors and compute uncertainty
        colors = torch.argmax(prediction, dim=1)  # example, x, y
        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)
        
        solution_slices = []
        uncertainty_values = []
        
        for example_num in range(self.task.n_test):
            # Determine output shape
            x_length = None
            y_length = None
            
            if self.task.in_out_same_size or self.task.all_out_same_size:
                shape_idx = self.task.n_train + example_num
                if shape_idx < len(self.task.shapes):
                    x_length = self.task.shapes[shape_idx][1][0]
                    y_length = self.task.shapes[shape_idx][1][1]
            
            # Extract solution with enhanced cropping
            solution_slice = self._enhanced_crop(
                colors[example_num], x_mask[example_num], x_length,
                y_mask[example_num], y_length
            )
            
            # Extract uncertainty
            uncertainty_slice = self._enhanced_crop(
                uncertainties[example_num], x_mask[example_num], x_length,
                y_mask[example_num], y_length
            )
            
            # Convert to list format
            solution_slices.append(solution_slice.cpu().numpy().tolist())
            uncertainty_values.append(float(uncertainty_slice.mean().cpu().numpy()))
        
        # Convert colors to task color indices
        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    if val < len(self.task.colors):
                        row[i] = self.task.colors[val]
                    else:
                        row[i] = 0  # Default to background
        
        # Create solution tuple
        solution_tuple = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        avg_uncertainty = np.mean(uncertainty_values)
        
        return solution_tuple, avg_uncertainty
    
    def _enhanced_crop(self, tensor: torch.Tensor, mask: torch.Tensor, 
                      target_length: Optional[int]) -> torch.Tensor:
        """Enhanced cropping with better boundary detection."""
        
        if target_length is not None:
            # Use known target length
            return tensor[:target_length]
        
        # Use mask-based cropping with confidence weighting
        if mask.dim() == 1:
            # 1D mask
            confidence_scores = F.softmax(mask, dim=0)
            
            # Find optimal crop based on confidence
            best_score = -float('inf')
            best_start, best_end = 0, len(mask)
            
            for length in range(1, len(mask) + 1):
                for start in range(len(mask) - length + 1):
                    end = start + length
                    score = (torch.sum(confidence_scores[start:end]) - 
                            torch.sum(confidence_scores[:start]) - 
                            torch.sum(confidence_scores[end:]))
                    
                    if score > best_score:
                        best_score = score
                        best_start, best_end = start, end
            
            return tensor[best_start:best_end]
        
        # Fallback to original tensor
        return tensor
    
    def get_ensemble_solutions(self, n_solutions: int = 4) -> List[Tuple]:
        """Get ensemble of diverse, high-quality solutions."""
        
        solutions = []
        
        # Add best solutions from different criteria
        solutions.extend([
            self.solution_most_frequent,
            self.solution_second_most_frequent,
            self.solution_highest_confidence,
            self.solution_most_consistent
        ])
        
        # Add diversity-weighted solutions from tracker
        if hasattr(self.solution_tracker, 'get_diversity_weighted_solutions'):
            diverse_solutions = self.solution_tracker.get_diversity_weighted_solutions(n_solutions)
            solutions.extend(diverse_solutions)
        
        # Remove duplicates while preserving order
        unique_solutions = []
        seen_hashes = set()
        
        for solution in solutions:
            sol_hash = hash(solution)
            if sol_hash not in seen_hashes:
                unique_solutions.append(solution)
                seen_hashes.add(sol_hash)
        
        # Ensure we have enough solutions
        while len(unique_solutions) < n_solutions:
            unique_solutions.append(self.solution_most_frequent)
        
        return unique_solutions[:n_solutions]
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        
        summary = {
            'total_steps': self.current_step,
            'final_loss': self.metrics_history['loss_curve'][-1] if self.metrics_history['loss_curve'] else 0,
            'best_loss': min(self.metrics_history['loss_curve']) if self.metrics_history['loss_curve'] else 0,
            'solutions_explored': len(self.solution_tracker.solutions),
            'unique_solutions': len(self.solution_hashes_count),
            'best_solution_confidence': max([q.confidence for q in self.solution_tracker.quality_scores]) if self.solution_tracker.quality_scores else 0,
            'convergence_step': self._estimate_convergence_step(),
        }
        
        return summary
    
    def _estimate_convergence_step(self) -> int:
        """Estimate when training converged."""
        if len(self.metrics_history['loss_curve']) < 100:
            return self.current_step
        
        # Look for when loss stopped decreasing significantly
        losses = self.metrics_history['loss_curve']
        window_size = 50
        
        for i in range(window_size, len(losses)):
            recent_trend = np.polyfit(range(window_size), losses[i-window_size:i], 1)[0]
            if abs(recent_trend) < 1e-4:
                return i
        
        return self.current_step


# Utility functions for analysis and saving
def save_predictions(loggers: List[EnhancedLogger], fname: str = 'predictions.npz'):
    """Save enhanced prediction data."""
    
    solution_contribution_logs = []
    solution_picks_histories = []
    training_summaries = []
    
    for logger in loggers:
        solution_contribution_logs.append(logger.solution_contributions_log)
        solution_picks_histories.append(logger.solution_picks_history)
        training_summaries.append(logger.get_training_summary())
    
    np.savez(
        fname,
        solution_contribution_logs=solution_contribution_logs,
        solution_picks_histories=solution_picks_histories,
        training_summaries=training_summaries
    )


def plot_enhanced_accuracy(true_solution_hashes: List[int], fname: str = 'predictions.npz'):
    """Plot enhanced accuracy metrics."""
    
    try:
        stored_data = np.load(fname, allow_pickle=True)
        solution_picks_histories = stored_data['solution_picks_histories']
        
        n_tasks = len(solution_picks_histories)
        n_iterations = len(solution_picks_histories[0]) if n_tasks > 0 else 0
        
        if n_iterations == 0:
            print("No data to plot")
            return
        
        # Calculate accuracy for different solution types
        accuracy_metrics = {}
        solution_types = ['most_frequent', 'second_frequent', 'highest_confidence', 'most_consistent']
        
        for sol_type_idx, sol_type in enumerate(solution_types):
            correct = np.array([[
                int(picks[sol_type_idx] == true_solution_hashes[task_num])
                for picks in task_history
            ] for task_num, task_history in enumerate(solution_picks_histories)])
            
            accuracy_metrics[sol_type] = correct.mean(axis=0)
        
        # Create enhanced plot
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy curves
        plt.subplot(2, 2, 1)
        for sol_type, accuracy_curve in accuracy_metrics.items():
            plt.plot(np.arange(n_iterations), accuracy_curve, label=sol_type, linewidth=2)
        
        plt.xlabel('Training Step')
        plt.ylabel('Accuracy')
        plt.title('Solution Accuracy by Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot convergence analysis
        plt.subplot(2, 2, 2)
        overall_accuracy = np.mean(list(accuracy_metrics.values()), axis=0)
        plt.plot(np.arange(n_iterations), overall_accuracy, 'k-', linewidth=3, label='Overall')
        plt.fill_between(np.arange(n_iterations), 
                        overall_accuracy - np.std(list(accuracy_metrics.values()), axis=0),
                        overall_accuracy + np.std(list(accuracy_metrics.values()), axis=0),
                        alpha=0.3)
        
        plt.xlabel('Training Step')
        plt.ylabel('Overall Accuracy')
        plt.title('Accuracy Convergence')
        plt.grid(True, alpha=0.3)
        
        # Plot final accuracy distribution
        plt.subplot(2, 2, 3)
        final_accuracies = [acc[-1] for acc in accuracy_metrics.values()]
        plt.bar(solution_types, final_accuracies, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
        plt.ylabel('Final Accuracy')
        plt.title('Final Accuracy by Solution Type')
        plt.xticks(rotation=45)
        
        # Plot training summaries if available
        plt.subplot(2, 2, 4)
        if 'training_summaries' in stored_data:
            summaries = stored_data['training_summaries']
            convergence_steps = [s.get('convergence_step', 0) for s in summaries]
            plt.hist(convergence_steps, bins=20, alpha=0.7, color='purple')
            plt.xlabel('Convergence Step')
            plt.ylabel('Number of Tasks')
            plt.title('Training Convergence Distribution')
        
        plt.tight_layout()
        plt.savefig('enhanced_accuracy_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced accuracy analysis saved to enhanced_accuracy_analysis.pdf")
        
    except Exception as e:
        print(f"Error plotting enhanced accuracy: {e}")


# Backward compatibility
class Logger(EnhancedLogger):
    """Backward compatible Logger class."""
    
    def __init__(self, task):
        super().__init__(task, config={})


def plot_accuracy(true_solution_hashes: List[int], fname: str = 'predictions.npz'):
    """Backward compatible accuracy plotting."""
    plot_enhanced_accuracy(true_solution_hashes, fname)