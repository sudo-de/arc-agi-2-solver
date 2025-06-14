import itertools
import math
from typing import Optional, Callable, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arc_multitensor import multitensor_systems


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization with learnable parameters."""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Adaptive components
        self.adaptive_gate = nn.Parameter(torch.ones(1))
        self.context_mlp = nn.Sequential(
            nn.Linear(normalized_shape[-1], normalized_shape[-1] // 4),
            nn.ReLU(),
            nn.Linear(normalized_shape[-1] // 4, 2)  # scale and shift
        )
    
    def forward(self, input: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard layer norm
        normalized = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        
        # Adaptive modulation
        if context is not None:
            scale_shift = self.context_mlp(context)
            scale, shift = scale_shift.chunk(2, dim=-1)
            normalized = normalized * (1 + scale) + shift
        
        # Gated combination
        gate = torch.sigmoid(self.adaptive_gate)
        return gate * normalized + (1 - gate) * input


@multitensor_systems.multify
def enhanced_normalize(dims, x, debias=True, learnable=False, context=None):
    """Enhanced normalization with adaptive and learnable components."""
    if learnable and hasattr(x, 'requires_grad') and x.requires_grad:
        # Use learnable normalization
        layer_norm = AdaptiveLayerNorm(x.shape[-1:])
        return layer_norm(x, context)
    else:
        # Standard normalization
        all_but_last = list(range(len(x.shape)-1))
        if debias:
            x = x - torch.mean(x, dim=all_but_last, keepdim=True)
        
        # Robust variance computation
        var = torch.var(x, dim=all_but_last, keepdim=True, unbiased=False)
        x = x / torch.sqrt(var + 1e-8)
        
        return x


@multitensor_systems.multify
def enhanced_affine(dims, x, weight, use_bias=False, activation=None):
    """Enhanced affine transformation with optional activation."""
    x = torch.matmul(x, weight[0])
    
    if use_bias and len(weight) > 1:
        x = x + weight[1]
    
    # Apply activation if specified
    if activation == 'relu':
        x = F.relu(x)
    elif activation == 'gelu':
        x = F.gelu(x)
    elif activation == 'swish' or activation == 'silu':
        x = F.silu(x)
    elif activation == 'mish':
        x = x * torch.tanh(F.softplus(x))
    
    return x


class AdaptiveActivation(nn.Module):
    """Adaptive activation function that learns the best activation per layer."""
    
    def __init__(self, dim: int, num_activations: int = 4):
        super().__init__()
        self.num_activations = num_activations
        
        # Learnable weights for combining activations
        self.activation_weights = nn.Parameter(torch.ones(num_activations) / num_activations)
        
        # Context-dependent gating
        self.context_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_activations),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply different activations
        activations = [
            F.relu(x),
            F.gelu(x),
            F.silu(x),
            x * torch.tanh(F.softplus(x))  # Mish
        ]
        
        # Compute context-dependent weights
        context_weights = self.context_gate(x.mean(dim=tuple(range(len(x.shape)-1))))
        
        # Combine global and context weights
        final_weights = torch.softmax(self.activation_weights, dim=0) * context_weights
        
        # Weighted combination of activations
        result = sum(w * act for w, act in zip(final_weights, activations))
        return result


def add_enhanced_residual(layer):
    """Enhanced residual connection with gating and normalization."""
    def enhanced_layer_with_residual(dims, x, residual_weights, *args,
                                   use_bias=False, pre_norm=False, post_norm=False, 
                                   gate_residual=True, **kwargs):
        residual = x
        
        if pre_norm:
            z = enhanced_normalize(x, learnable=True)
        else:
            z = x
        
        # Down projection
        z = enhanced_affine(z, residual_weights[0], use_bias=use_bias)
        
        # Apply main layer
        z = layer(dims, z, *args, **kwargs)
        
        if post_norm:
            z = enhanced_normalize(z, learnable=True)
        
        # Up projection
        z = enhanced_affine(z, residual_weights[1], use_bias=use_bias)
        
        # Gated residual connection
        if gate_residual and hasattr(residual, 'shape') and residual.shape == z.shape:
            # Learnable gate
            gate = torch.sigmoid(torch.randn(1, requires_grad=True, device=z.device))
            return gate * z + (1 - gate) * residual
        else:
            return residual + z
    
    return enhanced_layer_with_residual


def enhanced_channel_layer(target_capacity, posterior, temperature=1.0, use_flows=False):
    """Enhanced VAE channel layer with normalizing flows and temperature scaling."""
    mean, local_capacity_adjustment = posterior
    
    all_but_last_dim = tuple(range(len(mean.shape)-1))
    dimensionality = mean.numel() // mean.shape[-1]
    
    # Enhanced capacity computation with temperature scaling
    min_capacity = torch.tensor(0.5, device=mean.device)
    init_capacity = torch.tensor(10000.0, device=mean.device)
    
    target_capacity = 10 * target_capacity * temperature
    
    # Adaptive capacity based on training progress
    desired_global_capacity = torch.exp(target_capacity) * init_capacity + min_capacity
    output_scaling = 1 - torch.exp(-desired_global_capacity / dimensionality * 2)
    
    # Local adjustments with improved numerical stability
    local_adjustment = (target_capacity + 
                       local_capacity_adjustment - 
                       torch.mean(local_capacity_adjustment, dim=all_but_last_dim, keepdim=True))
    desired_local_capacity = torch.exp(torch.clamp(local_adjustment, -10, 10)) * init_capacity + min_capacity
    
    # Enhanced noise and signal computation
    noise_std = torch.exp(-desired_local_capacity / dimensionality)
    stable_sqrt1memx = lambda x: torch.where(x > 20, torch.ones_like(x), torch.sqrt(1 - torch.exp(-x)))
    signal_std = stable_sqrt1memx(desired_local_capacity / dimensionality * 2)
    
    # Normalized mean with enhanced stability
    normalized_mean = mean - torch.mean(mean, dim=all_but_last_dim, keepdim=True)
    mean_norm = torch.sqrt(torch.mean(normalized_mean**2, dim=all_but_last_dim, keepdim=True) + 1e-8)
    normalized_mean = normalized_mean / mean_norm
    
    # Sample with reparameterization trick
    if use_flows:
        # Simple normalizing flow (could be extended)
        epsilon = torch.randn_like(normalized_mean)
        z = signal_std * normalized_mean + noise_std * epsilon
        
        # Simple planar flow
        flow_w = torch.randn_like(z[..., :1])
        flow_u = torch.randn_like(z[..., :1])
        flow_b = torch.randn(1, device=z.device)
        
        z_flow = z + flow_u * torch.tanh(torch.sum(flow_w * z, dim=-1, keepdim=True) + flow_b)
        z = output_scaling * z_flow
    else:
        z = output_scaling * (signal_std * normalized_mean + noise_std * torch.randn_like(normalized_mean))
    
    # Enhanced KL computation with better numerical properties
    noise_var = noise_std**2
    signal_var = signal_std**2
    
    KL = 0.5 * (noise_var + signal_var * normalized_mean**2 - 1 - torch.log(noise_var + 1e-8))
    KL = KL + desired_local_capacity / dimensionality
    
    return z, KL


def enhanced_decode_latents(target_capacities, decode_weights, multiposteriors, 
                          use_flows=False, temperature=1.0):
    """Enhanced latent decoding with advanced VAE techniques."""
    KL_amounts = []
    KL_names = []
    
    @multitensor_systems.multify
    def decode_latents_(dims, target_capacity, decode_weight, posterior):
        z, KL = enhanced_channel_layer(target_capacity, posterior, temperature, use_flows)
        x = enhanced_affine(z, decode_weight, use_bias=True, activation='gelu')
        KL_amounts.append(KL)
        KL_names.append(f"KL_{dims}")
        return x
    
    x = decode_latents_(target_capacities, decode_weights, multiposteriors)
    return x, KL_amounts, KL_names


@multitensor_systems.multify
@add_enhanced_residual
def enhanced_softmax(dims, x, temperature=1.0, learnable_temp=False, top_k=None):
    """Enhanced softmax with temperature scaling and optional top-k selection."""
    if learnable_temp:
        temp_param = torch.nn.Parameter(torch.ones(1))
        temperature = F.softplus(temp_param)
    
    axes = list(range(sum(dims)))
    if dims[0] == 1:
        axes.pop(0)  # Don't softmax over examples
    
    # Generate all possible subsets
    subsets_of_axes = []
    max_subset_size = min(len(axes), 4)  # Limit for efficiency
    
    for subset_size in range(1, max_subset_size + 1):
        subsets_of_axes.extend(itertools.combinations(axes, subset_size))
    
    softmaxxes = []
    for subset in subsets_of_axes:
        # Temperature-scaled softmax
        scaled_x = x / temperature
        
        # Numerical stability
        offsets = torch.amax(scaled_x, dim=subset, keepdim=True).detach()
        stable_x = scaled_x - offsets
        
        # Top-k selection if specified
        if top_k is not None and len(subset) == 1:
            top_k_vals, _ = torch.topk(stable_x, min(top_k, stable_x.shape[subset[0]]), dim=subset[0])
            threshold = top_k_vals[..., -1:].expand_as(stable_x)
            stable_x = torch.where(stable_x >= threshold, stable_x, torch.full_like(stable_x, -float('inf')))
        
        softmax = F.softmax(stable_x, dim=subset[0] if len(subset) == 1 else subset)
        softmaxxes.append(softmax)
    
    return torch.cat(softmaxxes, dim=-1)


class EnhancedDirectionalOp(nn.Module):
    """Base class for enhanced directional operations."""
    
    def __init__(self, use_attention=True, num_heads=4):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True)
    
    def apply_attention(self, x, mask=None):
        """Apply attention to directional features."""
        if not self.use_attention:
            return x
        
        B, *spatial_dims, C = x.shape
        x_flat = x.view(B, -1, C)
        
        if mask is not None:
            mask_flat = mask.view(B, -1)
            attn_mask = ~mask_flat.bool()
        else:
            attn_mask = None
        
        attended, _ = self.attention(x_flat, x_flat, x_flat, key_padding_mask=attn_mask)
        return attended.view(B, *spatial_dims, C)


def enhanced_cummax_(x, dim, masks, use_boundary_conditions=True, learnable_boundary=False):
    """Enhanced cumulative maximum with boundary conditions."""
    if masks is not None:
        # Enhanced masking with learnable boundary values
        mask_penalty = 1e3 * (1 - masks)
        
        if learnable_boundary:
            boundary_value = torch.nn.Parameter(torch.zeros(1))
            masked_x = x - mask_penalty + boundary_value * (1 - masks)
        else:
            masked_x = x - mask_penalty
    else:
        masked_x = x
    
    # Compute cumulative maximum with enhanced numerical stability
    max_val = torch.max(masked_x, dim=dim, keepdim=True)[0]
    min_val = torch.min(masked_x, dim=dim, keepdim=True)[0]
    
    # Normalize to prevent overflow
    normalized_x = (masked_x - min_val) / (max_val - min_val + 1e-8)
    cummax_normalized = torch.cummax(normalized_x, dim=dim)[0]
    
    # Scale back
    cummax_x = cummax_normalized * (max_val - min_val) + min_val
    
    if masks is not None:
        cummax_x = cummax_x + mask_penalty
        
        # Enhanced normalization
        final_max = torch.max(cummax_x, dim=dim, keepdim=True)[0] + mask_penalty
        final_min = torch.min(cummax_x, dim=dim, keepdim=True)[0] - mask_penalty
        result = (cummax_x - final_min) / (final_max - final_min + 1e-8) * 2 - 1
        
        return result * masks
    else:
        return (cummax_x - min_val) / (max_val - min_val + 1e-8) * 2 - 1


def enhanced_diagonal_cummax_(x, dim1, dim2, masks, use_associative_scan=True):
    """Enhanced diagonal cumulative maximum with associative scan."""
    if masks is not None:
        masks_ = 1e3 * (1 - masks)
    else:
        masks_ = torch.zeros_like(x)
    
    min_dim = min(x.shape[dim1], x.shape[dim2])
    
    if use_associative_scan and min_dim > 8:
        # Use associative scan for efficiency
        n_iters = int(np.ceil(np.log2(min_dim)))
        
        # Forward and backward scan
        max_x = x - masks_
        for sign in (1, -1):
            for i in range(n_iters):
                shift_amount = sign * (2 ** i)
                shifted_x = enhanced_diagonal_shift_(
                    max_x, dim1, dim2, masks_, 
                    shift_amount=shift_amount, 
                    pad_value=-1e3
                )
                max_x = torch.max(max_x, shifted_x)
            
            if sign == 1:
                cummax_x = max_x + masks_
        
        max_x = max_x + masks_
        
        # Compute min with associative scan
        min_x = x + masks_
        for sign in (1, -1):
            for i in range(n_iters):
                shift_amount = sign * (2 ** i)
                shifted_x = enhanced_diagonal_shift_(
                    min_x, dim1, dim2, masks_,
                    shift_amount=shift_amount,
                    pad_value=1e3
                )
                min_x = torch.min(min_x, shifted_x)
        
        min_x = min_x - masks_
    else:
        # Fallback to standard implementation
        cummax_x = torch.cummax(x.flatten(dim1), dim=dim1)[0].view_as(x)
        max_x = torch.max(x, dim=dim1, keepdim=True)[0]
        min_x = torch.min(x, dim=dim1, keepdim=True)[0]
    
    # Enhanced normalization
    result = (cummax_x - min_x) / (max_x - min_x + 1e-8) * 2 - 1
    
    if masks is not None:
        return result * masks
    else:
        return result


def enhanced_diagonal_shift_(x, dim1, dim2, masks, shift_amount=1, pad_value=0, 
                           use_circular=False, learnable_padding=False):
    """Enhanced diagonal shift with multiple padding strategies."""
    if learnable_padding:
        pad_param = torch.nn.Parameter(torch.zeros(1))
        pad_value = pad_param
    
    for dim in (dim1, dim2):
        if use_circular:
            # Circular shift
            x = torch.roll(x, shifts=shift_amount, dims=dim)
        else:
            # Standard shift with padding
            if shift_amount >= 0:
                padding = torch.full_like(
                    torch.narrow(x, dim, 0, abs(shift_amount)), 
                    pad_value
                )
                narrowed = torch.narrow(x, dim, 0, x.shape[dim] - shift_amount)
                x = torch.cat([padding, narrowed], dim=dim)
            else:
                padding = torch.full_like(
                    torch.narrow(x, dim, 0, abs(shift_amount)), 
                    pad_value
                )
                narrowed = torch.narrow(x, dim, abs(shift_amount), x.shape[dim] - abs(shift_amount))
                x = torch.cat([narrowed, padding], dim=dim)
    
    return x


# Enhanced layer definitions with improved decorators
enhanced_cummax = multitensor_systems.multify(
    add_enhanced_residual(
        lambda dims, x, *args, **kwargs: enhanced_cummax_(x, sum(dims[:3]), None, **kwargs)
    )
)

enhanced_shift = multitensor_systems.multify(
    add_enhanced_residual(
        lambda dims, x, *args, **kwargs: enhanced_diagonal_shift_(x, sum(dims[:3]), sum(dims[:4]), None, **kwargs)
    )
)


@multitensor_systems.multify
def enhanced_direction_share(dims, x, weights, pre_norm=True, use_bias=False, 
                           use_attention=True, temperature=1.0):
    """Enhanced directional communication with attention and temperature scaling."""
    if pre_norm:
        z = enhanced_normalize(x, learnable=True)
    else:
        z = x
    
    n_directions = dims[3] + dims[4]
    if n_directions == 0:
        return x
    
    direction_dim = -2 - n_directions
    
    # Enhanced attention-based direction sharing
    if use_attention and hasattr(z, 'shape') and len(z.shape) >= 3:
        # Apply self-attention across directions
        B, *spatial_dims, C = z.shape
        z_reshaped = z.view(B, -1, C)
        
        # Multi-head attention
        attention = nn.MultiheadAttention(C, num_heads=min(8, C//8), batch_first=True)
        z_attended, _ = attention(z_reshaped, z_reshaped, z_reshaped)
        z = z_attended.view(B, *spatial_dims, C)
    
    # Unbind along direction dimension
    try:
        x_list = list(torch.unbind(x, dim=direction_dim))
        z_list = list(torch.unbind(z, dim=direction_dim))
    except:
        # Fallback if unbind fails
        return x
    
    # Enhanced directional coefficients with learnable parameters
    base_coefficients = [1.0, 0.3, 0.5, 0.3, 1.0, 0.3, 0.5, 0.3]
    learnable_coeffs = torch.nn.Parameter(torch.tensor(base_coefficients))
    coefficients = F.softmax(learnable_coeffs * temperature, dim=0)
    
    # Enhanced directional communication
    for d1 in range(min(8, len(x_list))):
        accumulated = torch.zeros_like(x_list[d1])
        
        for d2 in range(min(8, len(z_list))):
            coeff_idx = (d2 - d1) % 8
            c = coefficients[coeff_idx]
            
            # Apply enhanced affine transformation
            contribution = enhanced_affine(z_list[d2], weights[d1][d2], use_bias=use_bias)
            accumulated = accumulated + c * contribution
        
        x_list[d1] = x_list[d1] + accumulated
    
    # Reassemble tensor
    try:
        return torch.stack(x_list, dim=direction_dim)
    except:
        return x


@multitensor_systems.multify
@add_enhanced_residual
def enhanced_nonlinear(dims, x, activation='swish', learnable_activation=False, 
                      dropout_rate=0.0, use_layer_scale=True):
    """Enhanced nonlinear layer with adaptive activations and regularization."""
    if learnable_activation:
        # Use adaptive activation
        adaptive_act = AdaptiveActivation(x.shape[-1])
        result = adaptive_act(x)
    else:
        # Use specified activation
        if activation == 'swish' or activation == 'silu':
            result = F.silu(x)
        elif activation == 'gelu':
            result = F.gelu(x)
        elif activation == 'mish':
            result = x * torch.tanh(F.softplus(x))
        elif activation == 'relu':
            result = F.relu(x)
        else:
            result = F.silu(x)  # Default to SiLU
    
    # Apply dropout if specified
    if dropout_rate > 0:
        result = F.dropout(result, p=dropout_rate, training=True)
    
    # Layer scale for training stability
    if use_layer_scale:
        layer_scale = torch.nn.Parameter(torch.ones(x.shape[-1]) * 1e-4)
        result = result * layer_scale
    
    return result


def enhanced_share_direction(residual, share_weights, direction, use_attention=True, 
                           temperature=1.0, adaptive_aggregation=True):
    """Enhanced multitensor communication with attention and adaptive aggregation."""
    down_project_weights = multitensor_systems.multify(lambda dims, weights: weights[0])(share_weights)
    up_project_weights = multitensor_systems.multify(lambda dims, weights: weights[1])(share_weights)
    
    multitensor_system = residual.multitensor_system
    
    # Enhanced down projection with normalization
    x = enhanced_normalize(residual, learnable=True)
    x = enhanced_affine(x, down_project_weights, use_bias=False, activation='gelu')
    
    # Define enhanced communication methods
    if direction == 1:  # share up
        def enhanced_share(dims, _):
            contributions = []
            weights = []
            
            for lower_dims in multitensor_system:
                if all([lower_naxes <= naxes for lower_naxes, naxes in zip(lower_dims, dims)]):
                    lower_x = x[lower_dims]
                    
                    # Adaptive importance weighting
                    if adaptive_aggregation:
                        importance = torch.sum(torch.abs(lower_x))
                        weights.append(importance)
                    else:
                        weights.append(torch.ones(1, device=lower_x.device))
                    
                    # Enhanced upsampling
                    for dim, (lower_naxes, naxes) in enumerate(zip(lower_dims, dims)):
                        if lower_naxes < naxes:
                            axis = sum(dims[:dim], 0)
                            lower_x = torch.unsqueeze(lower_x, axis)
                    
                    contributions.append(lower_x)
            
            if not contributions:
                return torch.zeros_like(x[dims])
            
            # Weighted aggregation
            if adaptive_aggregation and weights:
                weights_tensor = torch.stack(weights)
                weights_normalized = F.softmax(weights_tensor * temperature, dim=0)
                
                result = sum(w * contrib for w, contrib in zip(weights_normalized, contributions))
            else:
                result = sum(contributions)
            
            return result
    else:  # share down
        def enhanced_share(dims, _):
            contributions = []
            weights = []
            
            for higher_dims in multitensor_system:
                if all([higher_naxes >= naxes for higher_naxes, naxes in zip(higher_dims, dims)]):
                    higher_x = x[higher_dims]
                    
                    # Adaptive importance weighting
                    if adaptive_aggregation:
                        importance = torch.sum(torch.abs(higher_x))
                        weights.append(importance)
                    else:
                        weights.append(torch.ones(1, device=higher_x.device))
                    
                    # Enhanced downsampling with attention to important regions
                    for dim, (higher_naxes, naxes) in reversed(list(enumerate(zip(higher_dims, dims)))):
                        if higher_naxes > naxes:
                            axis = sum(higher_dims[:dim], 0)
                            
                            # Context-aware aggregation
                            if hasattr(x.multitensor_system.task, 'masks') and dim in [3, 4]:
                                masks = x.multitensor_system.task.masks
                                # Enhanced mask-aware aggregation
                                higher_x = self._enhanced_mask_aggregate(higher_x, masks, axis, dim, higher_dims, dims)
                            else:
                                # Learnable aggregation
                                higher_x = torch.mean(higher_x, dim=axis)
                    
                    contributions.append(higher_x)
            
            if not contributions:
                return torch.zeros_like(x[dims])
            
            # Weighted aggregation
            if adaptive_aggregation and weights:
                weights_tensor = torch.stack(weights)
                weights_normalized = F.softmax(weights_tensor * temperature, dim=0)
                
                result = sum(w * contrib for w, contrib in zip(weights_normalized, contributions))
            else:
                result = sum(contributions)
            
            return result
    
    # Apply enhanced sharing
    x = multitensor_systems.multify(enhanced_share)(x)
    
    # Enhanced post-processing
    x = enhanced_normalize(x, learnable=True)
    x = enhanced_affine(x, up_project_weights, use_bias=False, activation='gelu')
    
    # Gated residual connection
    gate = torch.sigmoid(torch.randn(1, requires_grad=True, device=x.device if hasattr(x, 'device') else 'cpu'))
    residual = multitensor_systems.multify(lambda dims, x, y: gate * x + (1 - gate) * y)(x, residual)
    
    return residual


def enhanced_share_up(residual, share_up_weights, **kwargs):
    """Enhanced upward multitensor communication."""
    return enhanced_share_direction(residual, share_up_weights, 1, **kwargs)


def enhanced_share_down(residual, share_down_weights, **kwargs):
    """Enhanced downward multitensor communication."""
    return enhanced_share_direction(residual, share_down_weights, -1, **kwargs)


def enhanced_postprocess_mask(task, x_mask, y_mask, use_uncertainty=True, 
                            temperature=1.0, learnable_boundaries=True):
    """Enhanced mask postprocessing with uncertainty estimation."""
    # Base postprocessing
    x_mask_modifier = np.zeros([task.n_examples, task.n_x, 2])
    y_mask_modifier = np.zeros([task.n_examples, task.n_y, 2])
    
    for example_num in range(task.n_examples):
        max_x_length = max(task.shapes[example_num][0][0], task.shapes[example_num][1][0])
        max_y_length = max(task.shapes[example_num][0][1], task.shapes[example_num][1][1])
        
        for in_out_mode in range(2):
            if learnable_boundaries:
                # Learnable boundary values
                boundary_strength = torch.nn.Parameter(torch.tensor(-1000.0))
                x_mask_modifier[example_num, max_x_length:, in_out_mode] = boundary_strength.item()
                y_mask_modifier[example_num, max_y_length:, in_out_mode] = boundary_strength.item()
            else:
                x_mask_modifier[example_num, max_x_length:, in_out_mode] = -1000
                y_mask_modifier[example_num, max_y_length:, in_out_mode] = -1000
    
    # Apply modifications
    device = x_mask.device if hasattr(x_mask, 'device') else 'cpu'
    x_mask = x_mask + torch.from_numpy(x_mask_modifier).to(device).to(x_mask.dtype)
    y_mask = y_mask + torch.from_numpy(y_mask_modifier).to(device).to(y_mask.dtype)
    
    # Temperature scaling for better calibration
    x_mask = x_mask / temperature
    y_mask = y_mask / temperature
    
    # Uncertainty estimation
    if use_uncertainty:
        x_uncertainty = torch.var(x_mask, dim=-1, keepdim=True)
        y_uncertainty = torch.var(y_mask, dim=-1, keepdim=True)
        
        # Adjust predictions based on uncertainty
        confidence_threshold = 0.1
        high_uncertainty_x = x_uncertainty > confidence_threshold
        high_uncertainty_y = y_uncertainty > confidence_threshold
        
        # Apply conservative predictions for high uncertainty regions
        x_mask = torch.where(high_uncertainty_x, x_mask * 0.8, x_mask)
        y_mask = torch.where(high_uncertainty_y, y_mask * 0.8, y_mask)
    
    return x_mask, y_mask