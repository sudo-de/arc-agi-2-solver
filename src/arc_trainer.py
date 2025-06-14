import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

from .arc_network_layers import layers


class AdamW(Optimizer):
    """Enhanced AdamW optimizer with better numerical stability."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, maximize=False,
                 foreach=None, capturable=False, differentiable=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, maximize=maximize, foreach=foreach,
                       capturable=capturable, differentiable=differentiable)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.dtype in {torch.float16, torch.bfloat16}:
                        grads.append(p.grad.float())
                    else:
                        grads.append(p.grad)
                    
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    
                    state['step'] += 1
                    state_steps.append(state['step'])
            
            self._adamw_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=group['betas'][0],
                beta2=group['betas'][1],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize']
            )
        
        return loss
    
    def _adamw_step(self, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                   state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize):
        """Enhanced AdamW step with better numerical stability."""
        
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)
            
            # Bias correction
            step_size = lr / (1 - beta1 ** step)
            
            param.addcdiv_(exp_avg, denom, value=-step_size)


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Enhanced Cosine Annealing with Warm Restarts and adaptive parameters."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, eta_min_decay=1.0, 
                 restart_decay=1.0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_min_decay = eta_min_decay
        self.restart_decay = restart_decay
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0
        
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
                # Apply decay factors
                self.eta_min *= self.eta_min_decay
                self.base_lrs = [lr * self.restart_decay for lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = math.floor(epoch)
        
        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o
            
            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self
            
            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
        
        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class GradientClipping:
    """Advanced gradient clipping with adaptive thresholds."""
    
    def __init__(self, max_norm=1.0, norm_type=2.0, adaptive=True, percentile=95.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.adaptive = adaptive
        self.percentile = percentile
        self.grad_norms_history = []
        self.adaptive_threshold = max_norm
    
    def __call__(self, parameters):
        """Apply gradient clipping to parameters."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        
        if len(parameters) == 0:
            return torch.tensor(0.)
        
        device = parameters[0].grad.device
        
        if self.norm_type == 'inf':
            norms = [p.grad.detach().abs().max().to(device) for p in parameters]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), self.norm_type).to(device) 
                                               for p in parameters]), self.norm_type)
        
        # Update adaptive threshold
        if self.adaptive:
            self.grad_norms_history.append(total_norm.item())
            if len(self.grad_norms_history) > 1000:  # Keep last 1000 norms
                self.grad_norms_history = self.grad_norms_history[-1000:]
            
            if len(self.grad_norms_history) >= 10:
                self.adaptive_threshold = np.percentile(self.grad_norms_history, self.percentile)
                self.adaptive_threshold = max(self.adaptive_threshold, self.max_norm)
        
        # Apply clipping
        clip_coef = min(self.adaptive_threshold / (total_norm + 1e-6), 1.0)
        
        if clip_coef < 1.0:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        return total_norm


class AdvancedLoss:
    """Advanced loss computation with multiple components."""
    
    def __init__(self, 
                 kl_weight=1.0, 
                 reconstruction_weight=10.0,
                 uncertainty_weight=0.1,
                 consistency_weight=0.5,
                 adaptive_weights=True):
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.uncertainty_weight = uncertainty_weight
        self.consistency_weight = consistency_weight
        self.adaptive_weights = adaptive_weights
        
        # Adaptive weight tracking
        self.loss_history = {
            'kl': [],
            'reconstruction': [],
            'uncertainty': [],
            'consistency': []
        }
    
    def compute_enhanced_reconstruction_loss(self, logits, target, masks, example_shapes, 
                                           train_step, grid_size_uncertain_coeff=0.01):
        """Enhanced reconstruction loss with adaptive components."""
        reconstruction_error = 0
        
        for example_num in range(target.shape[0]):
            for in_out_mode in range(2):
                if example_num >= len(example_shapes) // 2 and in_out_mode == 1:
                    continue  # Skip test outputs
                
                # Determine grid size uncertainty
                grid_size_uncertain = not (
                    hasattr(self, 'task') and (
                        self.task.in_out_same_size or 
                        (self.task.all_out_same_size and in_out_mode == 1) or
                        (self.task.all_in_same_size and in_out_mode == 0)
                    )
                )
                
                if grid_size_uncertain:
                    coefficient = grid_size_uncertain_coeff ** max(0, 1 - train_step / 100)
                else:
                    coefficient = 1.0
                
                # Enhanced mask-aware loss computation
                logits_slice = logits[example_num, :, :, :, in_out_mode]
                target_slice = target[example_num, :, :, in_out_mode]
                
                if example_num < len(example_shapes):
                    output_shape = example_shapes[example_num][in_out_mode]
                    
                    # Enhanced cross-entropy with label smoothing
                    ce_loss = self._compute_label_smoothed_cross_entropy(
                        logits_slice, target_slice, output_shape, smoothing=0.1
                    )
                    
                    # Focal loss component for hard examples
                    focal_loss = self._compute_focal_loss(
                        logits_slice, target_slice, alpha=0.25, gamma=2.0
                    )
                    
                    # Combine losses
                    example_loss = 0.8 * ce_loss + 0.2 * focal_loss
                    reconstruction_error += coefficient * example_loss
        
        return reconstruction_error
    
    def _compute_label_smoothed_cross_entropy(self, logits, target, output_shape, smoothing=0.1):
        """Compute label smoothed cross-entropy loss."""
        log_probs = F.log_softmax(logits, dim=0)
        
        # Create one-hot targets with label smoothing
        num_classes = logits.shape[0]
        smooth_target = torch.full_like(log_probs, smoothing / (num_classes - 1))
        smooth_target.scatter_(0, target.unsqueeze(0), 1.0 - smoothing)
        
        # Compute loss only for valid regions
        valid_mask = torch.zeros_like(target, dtype=torch.bool)
        valid_mask[:output_shape[0], :output_shape[1]] = True
        
        loss = -torch.sum(smooth_target * log_probs * valid_mask.unsqueeze(0))
        loss = loss / valid_mask.sum()
        
        return loss
    
    def _compute_focal_loss(self, logits, target, alpha=0.25, gamma=2.0):
        """Compute focal loss for handling hard examples."""
        ce_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def compute_uncertainty_loss(self, logits, target):
        """Compute uncertainty-based loss for calibration."""
        # Compute predictive entropy
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Compute confidence
        max_probs = torch.max(probs, dim=1)[0]
        confidence = max_probs
        
        # Uncertainty loss: encourage low entropy for correct predictions
        correct_predictions = (torch.argmax(logits, dim=1) == target).float()
        uncertainty_loss = torch.mean(entropy * correct_predictions)
        
        return uncertainty_loss
    
    def compute_consistency_loss(self, logits_1, logits_2):
        """Compute consistency loss between different model outputs."""
        # KL divergence between predictions
        log_probs_1 = F.log_softmax(logits_1, dim=1)
        probs_2 = F.softmax(logits_2, dim=1)
        
        kl_loss = F.kl_div(log_probs_1, probs_2, reduction='batchmean')
        return kl_loss
    
    def update_adaptive_weights(self, losses_dict, train_step):
        """Update adaptive loss weights based on training progress."""
        if not self.adaptive_weights:
            return
        
        # Track loss history
        for key, value in losses_dict.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-100:]
        
        # Adaptive weight adjustment
        if train_step > 50 and all(len(hist) >= 10 for hist in self.loss_history.values()):
            # Compute relative loss scales
            avg_kl = np.mean(self.loss_history['kl'][-10:])
            avg_recon = np.mean(self.loss_history['reconstruction'][-10:])
            
            # Adjust weights to balance loss components
            if avg_recon > 10 * avg_kl:
                self.reconstruction_weight *= 0.99
                self.kl_weight *= 1.01
            elif avg_kl > 10 * avg_recon:
                self.kl_weight *= 0.99
                self.reconstruction_weight *= 1.01
            
            # Clamp weights
            self.kl_weight = max(0.1, min(5.0, self.kl_weight))
            self.reconstruction_weight = max(1.0, min(50.0, self.reconstruction_weight))


def enhanced_take_step(task, model, optimizer, train_step, train_history_logger, 
                      gradient_clipper=None, loss_computer=None, scheduler=None):
    """
    Enhanced training step with advanced optimization techniques.
    """
    if gradient_clipper is None:
        gradient_clipper = GradientClipping(max_norm=1.0, adaptive=True)
    
    if loss_computer is None:
        loss_computer = AdvancedLoss(adaptive_weights=True)
        loss_computer.task = task  # Attach task for grid size uncertainty
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
    
    # Add background color channel
    logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
    
    # Compute KL loss
    total_KL = sum(torch.sum(kl_amount) for kl_amount in KL_amounts)
    
    # Enhanced reconstruction loss
    reconstruction_error = loss_computer.compute_enhanced_reconstruction_loss(
        logits, task.problem, task.masks, task.shapes, train_step
    )
    
    # Additional loss components
    uncertainty_loss = loss_computer.compute_uncertainty_loss(
        logits.view(-1, logits.shape[1]), 
        task.problem.view(-1)
    )
    
    # Combine losses with adaptive weights
    losses_dict = {
        'kl': total_KL.item(),
        'reconstruction': reconstruction_error.item(),
        'uncertainty': uncertainty_loss.item(),
        'consistency': 0.0  # Placeholder for consistency loss
    }
    
    # Update adaptive weights
    loss_computer.update_adaptive_weights(losses_dict, train_step)
    
    # Combined loss
    loss = (loss_computer.kl_weight * total_KL + 
            loss_computer.reconstruction_weight * reconstruction_error +
            loss_computer.uncertainty_weight * uncertainty_loss)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    grad_norm = gradient_clipper(model.parameters())
    
    # Optimizer step
    optimizer.step()
    
    # Scheduler step
    if scheduler is not None:
        scheduler.step()
    
    # Enhanced logging
    train_history_logger.log(
        train_step, logits, x_mask, y_mask, KL_amounts, KL_names,
        total_KL, reconstruction_error, loss
    )
    
    # Return additional metrics
    return {
        'total_loss': loss.item(),
        'kl_loss': total_KL.item(),
        'reconstruction_loss': reconstruction_error.item(),
        'uncertainty_loss': uncertainty_loss.item(),
        'grad_norm': grad_norm.item(),
        'lr': optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
    }


def create_optimizer_and_scheduler(model, config):
    """Create enhanced optimizer and scheduler configuration."""
    
    # Parameter groups with different learning rates
    param_groups = []
    
    # Attention parameters (if present)
    attention_params = []
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'attention' in name.lower():
            attention_params.append(param)
        elif 'head' in name.lower() or 'output' in name.lower():
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    # Different learning rates for different components
    if attention_params:
        param_groups.append({
            'params': attention_params,
            'lr': config.get('attention_lr', 0.001),
            'weight_decay': config.get('attention_wd', 0.01)
        })
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': config.get('head_lr', 0.01),
            'weight_decay': config.get('head_wd', 0.001)
        })
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': config.get('backbone_lr', 0.005),
            'weight_decay': config.get('backbone_wd', 0.01)
        })
    
    # Create optimizer
    optimizer = AdamW(
        param_groups if param_groups else model.parameters(),
        lr=config.get('base_lr', 0.01),
        betas=config.get('betas', (0.9, 0.999)),
        eps=config.get('eps', 1e-8),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Create scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('T_0', 100),
        T_mult=config.get('T_mult', 2),
        eta_min=config.get('eta_min', 1e-6),
        eta_min_decay=config.get('eta_min_decay', 0.9),
        restart_decay=config.get('restart_decay', 0.95)
    )
    
    return optimizer, scheduler


def train_with_advanced_techniques(task, model, config, logger):
    """
    Train model with advanced techniques for competition performance.
    """
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create advanced components
    gradient_clipper = GradientClipping(
        max_norm=config.get('max_grad_norm', 1.0),
        adaptive=config.get('adaptive_clipping', True)
    )
    
    loss_computer = AdvancedLoss(
        kl_weight=config.get('kl_weight', 1.0),
        reconstruction_weight=config.get('recon_weight', 10.0),
        adaptive_weights=config.get('adaptive_weights', True)
    )
    loss_computer.task = task
    
    # Training loop
    n_iterations = config.get('n_iterations', 2000)
    metrics_history = []
    
    for train_step in range(n_iterations):
        # Enhanced training step
        metrics = enhanced_take_step(
            task, model, optimizer, train_step, logger,
            gradient_clipper, loss_computer, scheduler
        )
        
        metrics_history.append(metrics)
        
        # Progressive learning rate adjustment
        if train_step > 0 and train_step % 500 == 0:
            # Check for convergence
            recent_losses = [m['total_loss'] for m in metrics_history[-50:]]
            if len(recent_losses) >= 50:
                loss_trend = np.polyfit(range(50), recent_losses, 1)[0]
                if abs(loss_trend) < 1e-4:  # Converged
                    print(f"Converged at step {train_step}")
                    break
        
        # Adaptive early stopping
        if train_step > 1000:
            recent_grad_norms = [m['grad_norm'] for m in metrics_history[-100:]]
            if np.mean(recent_grad_norms) < 1e-5:
                print(f"Gradient norm too small at step {train_step}, stopping")
                break
    
    return metrics_history


# Backward compatibility
def take_step(task, model, optimizer, train_step, train_history_logger):
    """Standard training step for backward compatibility."""
    return enhanced_take_step(
        task, model, optimizer, train_step, train_history_logger
    )