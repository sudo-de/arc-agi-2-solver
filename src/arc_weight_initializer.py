import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Dict, Any, Optional, Union
import math

from .arc_multitensor import multitensor_systems


class OrthogonalConstraint:
    """Orthogonal constraint for maintaining weight orthogonality during training."""
    
    def __init__(self, beta: float = 0.01):
        self.beta = beta
    
    def __call__(self, module: nn.Module):
        """Apply orthogonal constraint to module parameters."""
        if hasattr(module, 'weight') and module.weight.dim() >= 2:
            w = module.weight
            if w.shape[0] >= w.shape[1]:
                # More rows than columns
                wwt = torch.mm(w, w.t())
                identity = torch.eye(w.shape[0], device=w.device, dtype=w.dtype)
                orthogonal_loss = self.beta * torch.norm(wwt - identity) ** 2
            else:
                # More columns than rows
                wtw = torch.mm(w.t(), w)
                identity = torch.eye(w.shape[1], device=w.device, dtype=w.dtype)
                orthogonal_loss = self.beta * torch.norm(wtw - identity) ** 2
            
            # Add to module's orthogonal loss (for use in training)
            if not hasattr(module, 'orthogonal_loss'):
                module.orthogonal_loss = 0
            module.orthogonal_loss += orthogonal_loss


class AdaptiveInitialization:
    """Adaptive initialization based on layer type and task characteristics."""
    
    def __init__(self, task_features: Optional[Dict[str, Any]] = None):
        self.task_features = task_features or {}
        
class AdaptiveInitialization:
    """Adaptive initialization based on layer type and task characteristics."""
    
    def __init__(self, task_features: Optional[Dict[str, Any]] = None):
        self.task_features = task_features or {}
        
    def compute_fan_in_out(self, tensor: torch.Tensor) -> tuple:
        """Compute fan_in and fan_out for various tensor shapes."""
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        
        if dimensions == 2:  # Linear layer
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:  # Conv layers
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        
        return fan_in, fan_out
    
    def adaptive_gain(self, layer_type: str, activation: str = 'relu') -> float:
        """Compute adaptive gain based on layer type and activation."""
        base_gains = {
            'linear': 1.0,
            'conv': 1.0,
            'residual': 0.1,  # Smaller for residual connections
            'attention': 0.02,  # Very small for attention
            'embedding': 0.1,
        }
        
        activation_gains = {
            'relu': math.sqrt(2.0),
            'leaky_relu': math.sqrt(2.0 / (1 + 0.01**2)),
            'selu': 3/4,
            'gelu': 1.0,
            'swish': 1.0,
            'silu': 1.0,
            'tanh': 5/3,
            'sigmoid': 1.0,
        }
        
        # Task-adaptive scaling
        task_complexity = self.task_features.get('complexity', 1.0)
        adaptive_scale = min(2.0, max(0.5, task_complexity))
        
        base_gain = base_gains.get(layer_type, 1.0)
        activation_gain = activation_gains.get(activation, 1.0)
        
        return base_gain * activation_gain * adaptive_scale
    
    def layerwise_adaptive_lr_scale(self, layer_depth: int, total_layers: int) -> float:
        """Compute layer-wise learning rate scaling factor."""
        # LARS-like scaling for different layer depths
        depth_ratio = layer_depth / max(1, total_layers - 1)
        
        # Scale later layers less aggressively
        scale = 1.0 - 0.1 * depth_ratio
        return max(0.1, scale)


class EnhancedInitializer:
    """Enhanced initializer with advanced techniques for ARC tasks."""
    
    def __init__(self, multitensor_system, channel_dim_fn, use_adaptive_init=True, 
                 use_orthogonal_constraint=True, task_features=None):
        self.multitensor_system = multitensor_system
        self.channel_dim_fn = channel_dim_fn
        self.weights_list = []
        self.use_adaptive_init = use_adaptive_init
        self.use_orthogonal_constraint = use_orthogonal_constraint
        
        # Initialize adaptive components
        if use_adaptive_init:
            self.adaptive_init = AdaptiveInitialization(task_features)
        
        if use_orthogonal_constraint:
            self.orthogonal_constraint = OrthogonalConstraint()
        
        # Track initialization statistics
        self.init_stats = {
            'weights_initialized': 0,
            'avg_weight_norm': 0.0,
            'max_weight_norm': 0.0,
            'min_weight_norm': float('inf')
        }
    
    def enhanced_xavier_uniform_(self, tensor: torch.Tensor, gain: float = 1.0, 
                                layer_type: str = 'linear') -> torch.Tensor:
        """Enhanced Xavier initialization with adaptive scaling."""
        fan_in, fan_out = self.adaptive_init.compute_fan_in_out(tensor) if self.use_adaptive_init else (tensor.size(1), tensor.size(0))
        
        # Adaptive gain computation
        if self.use_adaptive_init:
            adaptive_gain = self.adaptive_init.adaptive_gain(layer_type)
            total_gain = gain * adaptive_gain
        else:
            total_gain = gain
        
        # Enhanced Xavier with better numerical properties
        std = total_gain * math.sqrt(2.0 / float(fan_in + fan_out))
        
        # Prevent extreme initializations
        std = max(1e-4, min(1.0, std))
        
        # Use truncated normal for better properties
        with torch.no_grad():
            tensor.uniform_(-std * math.sqrt(3), std * math.sqrt(3))
        
        return tensor
    
    def enhanced_kaiming_uniform_(self, tensor: torch.Tensor, a: float = 0, 
                                 mode: str = 'fan_in', nonlinearity: str = 'relu',
                                 layer_type: str = 'conv') -> torch.Tensor:
        """Enhanced Kaiming initialization."""
        fan_in, fan_out = self.adaptive_init.compute_fan_in_out(tensor) if self.use_adaptive_init else (tensor.size(1), tensor.size(0))
        
        if mode == 'fan_in':
            num_input_fmaps = fan_in
        elif mode == 'fan_out':
            num_input_fmaps = fan_out
        else:
            num_input_fmaps = (fan_in + fan_out) / 2
        
        # Adaptive gain
        if self.use_adaptive_init:
            gain = self.adaptive_init.adaptive_gain(layer_type, nonlinearity)
        else:
            gain = nn.init.calculate_gain(nonlinearity, a)
        
        std = gain / math.sqrt(num_input_fmaps)
        std = max(1e-4, min(1.0, std))  # Clamp for stability
        
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
        
        return tensor
    
    def orthogonal_init_(self, tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Enhanced orthogonal initialization."""
        if tensor.numel() < 2:
            raise ValueError("Only tensors with 2 or more elements are supported")
        
        rows = tensor.size(0)
        cols = tensor.numel() // rows
        flattened = tensor.new(rows, cols).normal_(0, 1)
        
        if rows < cols:
            flattened.t_()
        
        # QR decomposition
        q, r = torch.qr(flattened)
        
        # Make Q uniform
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        
        if rows < cols:
            q.t_()
        
        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(gain)
        
        return tensor
    
    def spectral_norm_init_(self, tensor: torch.Tensor, target_spectral_norm: float = 1.0) -> torch.Tensor:
        """Initialize with specific spectral norm."""
        if tensor.dim() < 2:
            return tensor
        
        # Compute current spectral norm
        u, s, v = torch.svd(tensor.view(tensor.size(0), -1))
        current_norm = s[0]
        
        # Scale to target spectral norm
        if current_norm > 1e-8:
            tensor.data.div_(current_norm / target_spectral_norm)
        
        return tensor
    
    def update_init_stats(self, tensor: torch.Tensor, name: str = ""):
        """Update initialization statistics."""
        if tensor.numel() == 0:
            return
        
        norm = torch.norm(tensor).item()
        self.init_stats['weights_initialized'] += 1
        self.init_stats['avg_weight_norm'] += norm
        self.init_stats['max_weight_norm'] = max(self.init_stats['max_weight_norm'], norm)
        self.init_stats['min_weight_norm'] = min(self.init_stats['min_weight_norm'], norm)
    
    def initialize_zeros(self, dims, shape):
        """Enhanced zero initialization with optional small noise."""
        if callable(shape):
            shape = shape(dims)
        
        zeros = torch.zeros(shape, requires_grad=True)
        
        # Add tiny amount of noise to break symmetry
        if self.use_adaptive_init:
            noise_std = 1e-6
            zeros.data.add_(torch.randn_like(zeros) * noise_std)
        
        self.weights_list.append(zeros)
        self.update_init_stats(zeros, f"zeros_{dims}")
        return zeros
    
    def initialize_linear(self, dims, shape):
        """Enhanced linear layer initialization."""
        if callable(shape):
            shape = shape(dims)
        
        n_in, n_out = shape
        if callable(n_in):
            n_in = n_in(dims)
        if callable(n_out):
            n_out = n_out(dims)
        
        # Choose initialization scheme based on layer characteristics
        weight = torch.empty(n_in, n_out)
        bias = torch.empty(n_out)
        
        # Enhanced initialization schemes
        if self.use_adaptive_init:
            # Use adaptive Xavier for most cases
            self.enhanced_xavier_uniform_(weight, layer_type='linear')
            
            # Bias initialization with small positive values for stability
            fan_in = n_in
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bias.uniform_(-bound, bound)
            
            # Small positive bias for activation layers
            if n_out <= 64:  # Likely an activation layer
                bias.data.add_(0.01)
        else:
            # Standard initialization
            scale = 1 / np.sqrt(n_in) if n_in > 0 else 1.0
            weight.data.uniform_(-scale, scale)
            bias.data.uniform_(-scale, scale)
        
        weight.requires_grad = True
        bias.requires_grad = True
        
        # Apply orthogonal constraint if enabled
        if self.use_orthogonal_constraint and min(n_in, n_out) >= 2:
            self.orthogonal_init_(weight)
        
        self.weights_list.extend([weight, bias])
        self.update_init_stats(weight, f"linear_weight_{dims}")
        self.update_init_stats(bias, f"linear_bias_{dims}")
        
        return [weight, bias]
    
    def initialize_residual(self, dims, n_in, n_out):
        """Enhanced residual connection initialization."""
        # Use smaller initialization for residual connections
        linear_1 = self.initialize_linear(dims, [self.channel_dim_fn, n_in])
        linear_2 = self.initialize_linear(dims, [n_out, self.channel_dim_fn])
        
        # Scale down residual path for training stability
        if self.use_adaptive_init:
            residual_scale = 0.1
            for weight_bias in [linear_1, linear_2]:
                weight_bias[0].data.mul_(residual_scale)
                weight_bias[1].data.mul_(residual_scale)
        
        return [linear_1, linear_2]
    
    def initialize_posterior(self, dims, channel_dim):
        """Enhanced posterior initialization for VAE."""
        if callable(channel_dim):
            channel_dim = channel_dim(dims)
        
        shape = self.multitensor_system.shape(dims, channel_dim)
        
        # Enhanced mean initialization
        if self.use_adaptive_init:
            # Use smaller initial values for better training dynamics
            mean = 0.001 * torch.randn(shape)
            
            # Add task-specific initialization
            if hasattr(self.multitensor_system, 'task'):
                task_complexity = getattr(self.multitensor_system.task, 'n_colors', 5)
                complexity_scale = min(1.0, task_complexity / 10.0)
                mean.mul_(complexity_scale)
        else:
            mean = 0.01 * torch.randn(shape)
        
        mean.requires_grad = True
        
        # Enhanced capacity adjustment initialization
        local_capacity_adjustment = self.initialize_zeros(dims, shape)
        
        # Add small random perturbation to break symmetry
        if self.use_adaptive_init:
            local_capacity_adjustment.data.add_(torch.randn_like(local_capacity_adjustment) * 1e-4)
        
        self.weights_list.append(mean)
        self.update_init_stats(mean, f"posterior_mean_{dims}")
        
        return [mean, local_capacity_adjustment]
    
    def initialize_direction_share(self, dims, _):
        """Enhanced direction sharing initialization."""
        channel_dim_fn = self.channel_dim_fn
        
        direction_weights = []
        for i in range(8):
            direction_row = []
            for j in range(8):
                weights = self.initialize_linear(dims, [channel_dim_fn, channel_dim_fn])
                
                # Enhanced initialization for direction sharing
                if self.use_adaptive_init:
                    # Scale based on direction relationship
                    direction_distance = min(abs(i - j), 8 - abs(i - j))  # Circular distance
                    distance_scale = max(0.1, 1.0 - direction_distance * 0.1)
                    
                    weights[0].data.mul_(distance_scale)
                    weights[1].data.mul_(distance_scale)
                
                direction_row.append(weights)
            direction_weights.append(direction_row)
        
        return direction_weights
    
    def initialize_enhanced_head(self):
        """Enhanced head initialization with symmetry."""
        dims = [1, 1, 0, 1, 1]
        head_weights = self.initialize_linear(dims, [self.channel_dim_fn(dims), 2])
        
        # Enhanced symmetry enforcement
        if self.use_adaptive_init:
            # Initialize with slight asymmetry that gets averaged
            w1 = torch.randn_like(head_weights[0][..., 0])
            w2 = torch.randn_like(head_weights[0][..., 0])
            symmetric_weight = (w1 + w2) / 2
            
            head_weights[0].requires_grad = False
            head_weights[0] = torch.stack([symmetric_weight, symmetric_weight], dim=-1)
            head_weights[0].requires_grad = True
        else:
            # Standard symmetry
            head_weights[0].requires_grad = False
            head_weights[0] = torch.stack([head_weights[0][..., 0]] * 2, dim=-1)
            head_weights[0].requires_grad = True
        
        # Update weights list
        self.weights_list[-2] = head_weights[0]
        return head_weights
    
    def initialize_attention_weights(self, d_model: int, num_heads: int):
        """Initialize attention mechanism weights."""
        head_dim = d_model // num_heads
        
        # Query, Key, Value projections
        q_weight = torch.empty(d_model, d_model)
        k_weight = torch.empty(d_model, d_model)
        v_weight = torch.empty(d_model, d_model)
        out_weight = torch.empty(d_model, d_model)
        
        # Enhanced initialization for attention
        if self.use_adaptive_init:
            # Use smaller initialization for attention stability
            attention_scale = 1.0 / math.sqrt(d_model)
            
            for weight in [q_weight, k_weight, v_weight]:
                self.enhanced_xavier_uniform_(weight, gain=attention_scale, layer_type='attention')
            
            # Output projection with even smaller scale
            self.enhanced_xavier_uniform_(out_weight, gain=attention_scale * 0.1, layer_type='attention')
        else:
            for weight in [q_weight, k_weight, v_weight, out_weight]:
                nn.init.xavier_uniform_(weight)
        
        # Set requires_grad
        for weight in [q_weight, k_weight, v_weight, out_weight]:
            weight.requires_grad = True
        
        self.weights_list.extend([q_weight, k_weight, v_weight, out_weight])
        
        return {
            'q_proj': q_weight,
            'k_proj': k_weight,
            'v_proj': v_weight,
            'out_proj': out_weight
        }
    
    def initialize_layer_scale(self, dim: int, init_value: float = 1e-4):
        """Initialize layer scale parameters for training stability."""
        layer_scale = torch.full((dim,), init_value, requires_grad=True)
        self.weights_list.append(layer_scale)
        return layer_scale
    
    def initialize_adaptive_temperature(self, init_temp: float = 1.0):
        """Initialize learnable temperature parameters."""
        temperature = torch.tensor(init_temp, requires_grad=True)
        self.weights_list.append(temperature)
        return temperature
    
    # Multitensor initialization methods
    def initialize_multizeros(self, shape):
        return multitensor_systems.multify(self.initialize_zeros)(
            self.multitensor_system.make_multitensor(default=shape)
        )
    
    def initialize_multilinear(self, shape):
        return multitensor_systems.multify(self.initialize_linear)(
            self.multitensor_system.make_multitensor(default=shape)
        )
    
    def initialize_multiresidual(self, n_in, n_out):
        return multitensor_systems.multify(self.initialize_residual)(
            n_in, self.multitensor_system.make_multitensor(default=n_out)
        )
    
    def initialize_multiposterior(self, decoding_dim):
        return multitensor_systems.multify(self.initialize_posterior)(
            self.multitensor_system.make_multitensor(default=decoding_dim)
        )
    
    def initialize_multidirection_share(self):
        return multitensor_systems.multify(self.initialize_direction_share)(
            self.multitensor_system.make_multitensor()
        )
    
    def symmetrize_xy(self, multiweights):
        """Enhanced xy symmetrization with better numerical properties."""
        for dims in self.multitensor_system:
            if dims[3] == 0 and dims[4] == 1:
                target_dims = dims[:3] + [1, 0]
                if tuple(target_dims) in multiweights.data:
                    source_weights = multiweights[tuple(target_dims)]
                    
                    # Enhanced symmetrization with interpolation for smoother transition
                    if self.use_adaptive_init and hasattr(source_weights, 'data'):
                        # Interpolate between original and symmetric weights
                        alpha = 0.9  # Interpolation factor
                        current_weights = multiweights[dims]
                        
                        if hasattr(current_weights, 'data') and hasattr(source_weights, 'data'):
                            multiweights[dims].data = (
                                alpha * source_weights.data + 
                                (1 - alpha) * current_weights.data
                            )
                    else:
                        multiweights[dims] = source_weights
    
    def symmetrize_direction_sharing(self, multiweights):
        """Enhanced direction sharing symmetrization."""
        for dims in self.multitensor_system:
            for dir1 in range(8):
                for dir2 in range(8):
                    from_dims = dims
                    from_dir1, from_dir2 = dir1, dir2
                    
                    # Enhanced symmetrization logic with better handling
                    if dims[3] + dims[4] == 1:
                        from_dims = dims[:3] + [1, 0]
                        if dims[4] == 1:
                            from_dir1 = (2 + from_dir1) % 8
                            from_dir2 = (2 + from_dir2) % 8
                        
                        # Enhanced direction mapping
                        if from_dir1 > 4 or (from_dir1 in {0, 4} and from_dir2 > 4):
                            from_dir1 = (8 - from_dir1) % 8
                            from_dir2 = (8 - from_dir2) % 8
                        
                        if 2 < from_dir1 < 6 or (from_dir1 in {2, 6} and 2 < from_dir2 < 6):
                            from_dir1 = (4 - from_dir1) % 8
                            from_dir2 = (4 - from_dir2) % 8
                    else:
                        # Rotational symmetry
                        rotation = (from_dir1 // 2) * 2
                        from_dir1 = (from_dir1 - rotation) % 8
                        from_dir2 = (from_dir2 - rotation) % 8
                        
                        if (from_dir2 - from_dir1) % 8 > 4:
                            from_dir2 = (8 + 2 * from_dir1 - from_dir2) % 8
                    
                    # Enhanced weight sharing with interpolation
                    try:
                        source_weights = multiweights[from_dims][from_dir1][from_dir2]
                        
                        if self.use_adaptive_init:
                            # Smooth interpolation for better training dynamics
                            current_weights = multiweights[dims][dir1][dir2]
                            
                            # Interpolate between current and symmetric weights
                            alpha = 0.95
                            if hasattr(current_weights, 'data') and hasattr(source_weights, 'data'):
                                for i, (curr_w, src_w) in enumerate(zip(current_weights, source_weights)):
                                    if hasattr(curr_w, 'data') and hasattr(src_w, 'data'):
                                        curr_w.data = alpha * src_w.data + (1 - alpha) * curr_w.data
                            else:
                                multiweights[dims][dir1][dir2] = source_weights
                        else:
                            multiweights[dims][dir1][dir2] = source_weights
                    except (KeyError, IndexError, TypeError):
                        # Handle cases where symmetrization fails gracefully
                        continue
    
    def apply_weight_regularization(self):
        """Apply various weight regularization techniques."""
        if not self.use_adaptive_init:
            return
        
        for weight in self.weights_list:
            if not hasattr(weight, 'data') or weight.numel() == 0:
                continue
            
            # Spectral normalization for large matrices
            if weight.dim() >= 2 and weight.shape[0] >= 32:
                self.spectral_norm_init_(weight, target_spectral_norm=1.0)
            
            # Gradient clipping preparation (add hooks)
            if weight.requires_grad:
                def clip_grad_hook(grad):
                    return torch.clamp(grad, -1.0, 1.0)
                
                weight.register_hook(clip_grad_hook)
    
    def finalize_initialization(self):
        """Finalize initialization with post-processing."""
        # Compute final statistics
        if self.init_stats['weights_initialized'] > 0:
            self.init_stats['avg_weight_norm'] /= self.init_stats['weights_initialized']
        
        # Apply final regularization
        self.apply_weight_regularization()
        
        # Print initialization summary
        print(f"Enhanced Initialization Summary:")
        print(f"  Weights initialized: {self.init_stats['weights_initialized']}")
        print(f"  Average weight norm: {self.init_stats['avg_weight_norm']:.4f}")
        print(f"  Max weight norm: {self.init_stats['max_weight_norm']:.4f}")
        print(f"  Min weight norm: {self.init_stats['min_weight_norm']:.4f}")
        
        return self.init_stats


# Factory function for creating initializers
def create_initializer(multitensor_system, channel_dim_fn, variant='enhanced', **kwargs):
    """
    Factory function for creating different initializer variants.
    
    Args:
        multitensor_system: The multitensor system
        channel_dim_fn: Channel dimension function
        variant: Initializer variant ('basic', 'enhanced', 'competition')
        **kwargs: Additional parameters
    
    Returns:
        Initializer instance
    """
    if variant == 'basic':
        from .initializers_basic import Initializer
        return Initializer(multitensor_system, channel_dim_fn)
    
    elif variant == 'enhanced':
        return EnhancedInitializer(
            multitensor_system, 
            channel_dim_fn,
            use_adaptive_init=True,
            use_orthogonal_constraint=True,
            **kwargs
        )
    
    elif variant == 'competition':
        # Competition-grade configuration
        task_features = kwargs.get('task_features', {})
        
        return EnhancedInitializer(
            multitensor_system,
            channel_dim_fn,
            use_adaptive_init=True,
            use_orthogonal_constraint=True,
            task_features=task_features,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown initializer variant: {variant}")