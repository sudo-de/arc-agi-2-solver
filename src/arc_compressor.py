import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math

from .arc_weight_initializer import initializers
from .arc_network_layers import layers
from .arc_multitensor import MultiTensorSystem


class AdaptiveArchitecture(nn.Module):
    """Adaptive architecture that adjusts based on task complexity."""
    
    def __init__(self, base_dim: int = 16, max_layers: int = 8):
        super().__init__()
        self.base_dim = base_dim
        self.max_layers = max_layers
        
        # Architecture controller
        self.complexity_estimator = nn.Sequential(
            nn.Linear(5, 32),  # 5 task features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, max_layers),
            nn.Sigmoid()
        )
        
        # Layer importance weights
        self.layer_gates = nn.Parameter(torch.ones(max_layers))
        
    def estimate_complexity(self, task_features: torch.Tensor) -> torch.Tensor:
        """Estimate task complexity and return layer weights."""
        return self.complexity_estimator(task_features)
    
    def get_active_layers(self, complexity_scores: torch.Tensor) -> List[int]:
        """Determine which layers should be active."""
        threshold = 0.5
        active = (complexity_scores * torch.sigmoid(self.layer_gates)) > threshold
        return torch.where(active)[0].tolist()


class AttentionBlock(nn.Module):
    """Multi-head attention for spatial and feature dimensions."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out)


class ProgressiveRefinement(nn.Module):
    """Progressive refinement with multiple resolution scales."""
    
    def __init__(self, base_dim: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # Multi-scale processing
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_dim, base_dim, 3, padding=1),
                nn.GroupNorm(8, base_dim),
                nn.SiLU(),
                nn.Conv2d(base_dim, base_dim, 3, padding=1),
            ) for _ in range(num_scales)
        ])
        
        # Cross-scale fusion
        self.fusion = nn.Conv2d(base_dim * num_scales, base_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Process at different scales
        scale_features = []
        for i, processor in enumerate(self.scale_processors):
            # Downsample
            scale_factor = 2 ** i
            if scale_factor > 1:
                x_scaled = F.avg_pool2d(x, scale_factor)
            else:
                x_scaled = x
            
            # Process
            feat = processor(x_scaled)
            
            # Upsample back
            if scale_factor > 1:
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            
            scale_features.append(feat)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_features, dim=1)
        return self.fusion(fused)


class AdvancedARCCompressor(nn.Module):
    """
    Advanced ARC Compressor with state-of-the-art techniques:
    - Adaptive architecture based on task complexity
    - Multi-head attention mechanisms
    - Progressive refinement at multiple scales
    - Ensemble of specialized sub-models
    - Advanced regularization and optimization
    """
    
    def __init__(self, task, config: Optional[Dict] = None):
        super().__init__()
        
        # Default configuration for top performance
        self.config = {
            'n_layers': 6,
            'base_dim': 32,
            'attention_heads': 8,
            'dropout': 0.1,
            'use_adaptive_arch': True,
            'use_attention': True,
            'use_progressive_refinement': True,
            'ensemble_size': 3,
            'max_adaptive_layers': 8,
        }
        if config:
            self.config.update(config)
        
        self.task = task
        self.multitensor_system = task.multitensor_system
        
        # Initialize components
        self._init_adaptive_architecture()
        self._init_core_model()
        self._init_attention_mechanisms()
        self._init_progressive_refinement()
        self._init_ensemble_components()
        self._init_weights()
        
    def _init_adaptive_architecture(self):
        """Initialize adaptive architecture components."""
        if self.config['use_adaptive_arch']:
            self.adaptive_arch = AdaptiveArchitecture(
                base_dim=self.config['base_dim'],
                max_layers=self.config['max_adaptive_layers']
            )
            
            # Task feature extractor
            self.task_feature_extractor = nn.Sequential(
                nn.Linear(10, 32),  # Basic task statistics
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 5)
            )
    
    def _init_core_model(self):
        """Initialize core VAE decoder model."""
        # Enhanced channel dimensions
        self.share_up_dim = self.config['base_dim'] * 2
        self.share_down_dim = self.config['base_dim']
        self.decoding_dim = self.config['base_dim'] // 2
        self.softmax_dim = 8
        self.cummax_dim = self.config['base_dim'] // 2
        self.shift_dim = self.config['base_dim'] // 2
        self.nonlinear_dim = self.config['base_dim']
        
        # Channel dimension function with adaptive sizing
        def enhanced_channel_dim_fn(dims):
            base = self.config['base_dim']
            if dims[2] == 0:  # No direction dimension
                return base * 2
            else:
                return base
        
        self.channel_dim_fn = enhanced_channel_dim_fn
        
        # Initialize enhanced weights
        initializer = initializers.EnhancedInitializer(
            self.multitensor_system, 
            self.channel_dim_fn,
            use_adaptive_init=True
        )
        
        # Core components with enhanced dimensions
        self.multiposteriors = initializer.initialize_multiposterior(self.decoding_dim)
        self.decode_weights = initializer.initialize_multilinear([self.decoding_dim, self.channel_dim_fn])
        self.target_capacities = initializer.initialize_multizeros([self.decoding_dim])
        
        # Enhanced layer weights
        self.share_up_weights = []
        self.share_down_weights = []
        self.softmax_weights = []
        self.cummax_weights = []
        self.shift_weights = []
        self.direction_share_weights = []
        self.nonlinear_weights = []
        
        for layer_num in range(self.config['n_layers']):
            self.share_up_weights.append(
                initializer.initialize_multiresidual(self.share_up_dim, self.share_up_dim)
            )
            self.share_down_weights.append(
                initializer.initialize_multiresidual(self.share_down_dim, self.share_down_dim)
            )
            
            # Enhanced softmax with adaptive output scaling
            output_scaling_fn = lambda dims: self.softmax_dim * (2 ** min(dims[1] + dims[2] + dims[3] + dims[4], 6))
            self.softmax_weights.append(
                initializer.initialize_multiresidual(self.softmax_dim, output_scaling_fn)
            )
            
            self.cummax_weights.append(
                initializer.initialize_multiresidual(self.cummax_dim, self.cummax_dim)
            )
            self.shift_weights.append(
                initializer.initialize_multiresidual(self.shift_dim, self.shift_dim)
            )
            self.direction_share_weights.append(
                initializer.initialize_multidirection_share()
            )
            self.nonlinear_weights.append(
                initializer.initialize_multiresidual(self.nonlinear_dim, self.nonlinear_dim)
            )
        
        # Enhanced head and mask weights
        self.head_weights = initializer.initialize_enhanced_head()
        self.mask_weights = initializer.initialize_linear(
            [1, 0, 0, 1, 0], 
            [self.channel_dim_fn([1, 0, 0, 1, 0]), 4]  # Enhanced mask channels
        )
        
        # Symmetrize weights
        self._symmetrize_weights(initializer)
        
        self.weights_list = initializer.weights_list
    
    def _init_attention_mechanisms(self):
        """Initialize attention mechanisms."""
        if self.config['use_attention']:
            # Spatial attention
            self.spatial_attention = AttentionBlock(
                dim=self.config['base_dim'],
                num_heads=self.config['attention_heads'],
                dropout=self.config['dropout']
            )
            
            # Feature attention
            self.feature_attention = AttentionBlock(
                dim=self.config['base_dim'],
                num_heads=self.config['attention_heads'] // 2,
                dropout=self.config['dropout']
            )
            
            # Cross-attention between input and output
            self.cross_attention = AttentionBlock(
                dim=self.config['base_dim'],
                num_heads=self.config['attention_heads'],
                dropout=self.config['dropout']
            )
    
    def _init_progressive_refinement(self):
        """Initialize progressive refinement components."""
        if self.config['use_progressive_refinement']:
            self.progressive_refiner = ProgressiveRefinement(
                base_dim=self.config['base_dim'],
                num_scales=3
            )
            
            # Refinement controller
            self.refinement_controller = nn.Sequential(
                nn.Linear(self.config['base_dim'], self.config['base_dim'] // 2),
                nn.ReLU(),
                nn.Linear(self.config['base_dim'] // 2, 1),
                nn.Sigmoid()
            )
    
    def _init_ensemble_components(self):
        """Initialize ensemble of specialized models."""
        self.ensemble_size = self.config['ensemble_size']
        
        # Specialized sub-models
        self.color_specialist = self._create_specialist_head('color')
        self.spatial_specialist = self._create_specialist_head('spatial')
        self.pattern_specialist = self._create_specialist_head('pattern')
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(self.ensemble_size) / self.ensemble_size)
        
        # Meta-learner for ensemble combination
        self.meta_learner = nn.Sequential(
            nn.Linear(self.config['base_dim'] * self.ensemble_size, self.config['base_dim']),
            nn.ReLU(),
            nn.Linear(self.config['base_dim'], self.ensemble_size),
            nn.Softmax(dim=-1)
        )
    
    def _create_specialist_head(self, specialist_type: str) -> nn.Module:
        """Create specialized head for different aspects."""
        if specialist_type == 'color':
            return nn.Sequential(
                nn.Linear(self.config['base_dim'], self.config['base_dim']),
                nn.ReLU(),
                nn.Linear(self.config['base_dim'], 10),  # 10 colors
            )
        elif specialist_type == 'spatial':
            return nn.Sequential(
                nn.Conv2d(self.config['base_dim'], self.config['base_dim'], 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.config['base_dim'], 2, 1),  # x, y masks
            )
        elif specialist_type == 'pattern':
            return nn.Sequential(
                nn.Linear(self.config['base_dim'], self.config['base_dim'] * 2),
                nn.ReLU(),
                nn.Linear(self.config['base_dim'] * 2, self.config['base_dim']),
            )
        else:
            raise ValueError(f"Unknown specialist type: {specialist_type}")
    
    def _init_weights(self):
        """Initialize weights with advanced techniques."""
        # Xavier/Glorot initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _symmetrize_weights(self, initializer):
        """Apply symmetrization to weights."""
        for weight_list in [
            self.share_up_weights,
            self.share_down_weights,
            self.softmax_weights,
            self.cummax_weights,
            self.shift_weights,
            self.nonlinear_weights,
        ]:
            for layer_num in range(self.config['n_layers']):
                initializer.symmetrize_xy(weight_list[layer_num])
        
        for layer_num in range(self.config['n_layers']):
            initializer.symmetrize_direction_sharing(self.direction_share_weights[layer_num])
    
    def extract_task_features(self) -> torch.Tensor:
        """Extract features that characterize the current task."""
        # Basic task statistics
        features = torch.tensor([
            self.task.n_examples,
            self.task.n_colors,
            self.task.n_x,
            self.task.n_y,
            len(self.task.colors),
            float(self.task.in_out_same_size),
            float(self.task.all_out_same_size),
            float(self.task.all_in_same_size),
            np.mean([shape[0][0] * shape[0][1] for shape in self.task.shapes]),  # avg input size
            np.mean([shape[1][0] * shape[1][1] if shape[1] else 0 for shape in self.task.shapes]),  # avg output size
        ], dtype=torch.float32, device=next(self.parameters()).device)
        
        return self.task_feature_extractor(features)
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[str]]:
        """
        Enhanced forward pass with adaptive architecture and attention.
        """
        # Extract task features for adaptive architecture
        if self.config['use_adaptive_arch']:
            task_features = self.extract_task_features()
            complexity_scores = self.adaptive_arch.estimate_complexity(task_features)
            active_layers = self.adaptive_arch.get_active_layers(complexity_scores)
        else:
            active_layers = list(range(self.config['n_layers']))
        
        # Decoding layer with enhanced KL tracking
        x, KL_amounts, KL_names = layers.enhanced_decode_latents(
            self.target_capacities, self.decode_weights, self.multiposteriors
        )
        
        # Main processing loop with adaptive layers
        for layer_num in range(self.config['n_layers']):
            if layer_num not in active_layers:
                continue
            
            # Multitensor communication layer (up)
            x = layers.enhanced_share_up(x, self.share_up_weights[layer_num])
            
            # Apply attention if enabled
            if self.config['use_attention'] and layer_num % 2 == 0:
                x = self._apply_attention_to_multitensor(x)
            
            # Enhanced softmax layer with learnable temperature
            x = layers.enhanced_softmax(
                x, 
                self.softmax_weights[layer_num], 
                pre_norm=True, 
                post_norm=False, 
                use_bias=False,
                temperature=1.0 + 0.1 * layer_num  # Learnable temperature
            )
            
            # Directional layers with progressive refinement
            x = layers.enhanced_cummax(
                x, 
                self.cummax_weights[layer_num], 
                self.multitensor_system.task.masks,
                pre_norm=False, 
                post_norm=True, 
                use_bias=False
            )
            
            x = layers.enhanced_shift(
                x, 
                self.shift_weights[layer_num], 
                self.multitensor_system.task.masks,
                pre_norm=False, 
                post_norm=True, 
                use_bias=False
            )
            
            # Enhanced directional communication
            x = layers.enhanced_direction_share(
                x, 
                self.direction_share_weights[layer_num], 
                pre_norm=True, 
                use_bias=False
            )
            
            # Enhanced nonlinear layer with adaptive activation
            x = layers.enhanced_nonlinear(
                x, 
                self.nonlinear_weights[layer_num], 
                pre_norm=True, 
                post_norm=False, 
                use_bias=False,
                activation='swish'  # Swish activation for better gradients
            )
            
            # Multitensor communication layer (down)
            x = layers.enhanced_share_down(x, self.share_down_weights[layer_num])
            
            # Progressive refinement
            if self.config['use_progressive_refinement'] and layer_num % 3 == 2:
                x = self._apply_progressive_refinement(x)
            
            # Enhanced normalization with learnable parameters
            x = layers.enhanced_normalize(x, learnable=True)
        
        # Ensemble output generation
        outputs = self._generate_ensemble_outputs(x)
        
        # Enhanced postprocessing
        output, x_mask, y_mask = self._enhanced_postprocessing(outputs)
        
        return output, x_mask, y_mask, KL_amounts, KL_names
    
    def _apply_attention_to_multitensor(self, x):
        """Apply attention mechanisms to multitensor."""
        # This is a simplified version - full implementation would handle multitensor structure
        if hasattr(x, 'data') and isinstance(x.data, dict):
            # Apply attention to each tensor in the multitensor
            for dims, tensor in x.data.items():
                if tensor.dim() >= 3:
                    # Reshape for attention
                    B, *spatial_dims, C = tensor.shape
                    tensor_flat = tensor.view(B, -1, C)
                    
                    # Apply spatial attention
                    attended = self.spatial_attention(tensor_flat)
                    
                    # Reshape back
                    x.data[dims] = attended.view(B, *spatial_dims, C)
        
        return x
    
    def _apply_progressive_refinement(self, x):
        """Apply progressive refinement to multitensor."""
        # Similar to attention, this would be applied to each tensor
        if hasattr(x, 'data') and isinstance(x.data, dict):
            for dims, tensor in x.data.items():
                if tensor.dim() == 4:  # B, C, H, W format
                    refined = self.progressive_refiner(tensor)
                    
                    # Blend with original based on refinement controller
                    blend_weight = self.refinement_controller(tensor.mean(dim=[2, 3]))
                    blend_weight = blend_weight.unsqueeze(-1).unsqueeze(-1)
                    
                    x.data[dims] = tensor * (1 - blend_weight) + refined * blend_weight
        
        return x
    
    def _generate_ensemble_outputs(self, x):
        """Generate outputs from ensemble of specialists."""
        # Extract features for each specialist
        base_features = self._extract_base_features(x)
        
        # Generate specialist outputs
        color_out = self.color_specialist(base_features)
        spatial_out = self.spatial_specialist(base_features.unsqueeze(-1).unsqueeze(-1))
        pattern_out = self.pattern_specialist(base_features)
        
        # Combine with meta-learner
        ensemble_features = torch.cat([color_out, spatial_out.flatten(2), pattern_out], dim=-1)
        ensemble_weights = self.meta_learner(ensemble_features)
        
        # Weighted combination
        final_output = (
            ensemble_weights[..., 0:1] * color_out +
            ensemble_weights[..., 1:2] * spatial_out.flatten(2) +
            ensemble_weights[..., 2:3] * pattern_out
        )
        
        return final_output
    
    def _extract_base_features(self, x):
        """Extract base features from multitensor for ensemble."""
        # This is a simplified extraction - full version would properly handle multitensor
        if hasattr(x, 'data') and isinstance(x.data, dict):
            # Use the main tensor configuration
            main_dims = [1, 1, 0, 1, 1]  # examples, colors, x, y
            if tuple(main_dims) in x.data:
                return x.data[tuple(main_dims)]
        
        # Fallback to first available tensor
        return torch.randn(1, self.config['base_dim'], device=next(self.parameters()).device)
    
    def _enhanced_postprocessing(self, outputs):
        """Enhanced postprocessing with uncertainty estimation."""
        # Main output head with enhanced processing
        output = (
            layers.enhanced_affine(outputs, self.head_weights, use_bias=False) +
            100 * self.head_weights[1]
        )
        
        # Enhanced mask prediction with uncertainty
        x_mask = layers.enhanced_affine(outputs, self.mask_weights, use_bias=True)
        y_mask = layers.enhanced_affine(outputs, self.mask_weights, use_bias=True)
        
        # Apply enhanced postprocessing
        x_mask, y_mask = layers.enhanced_postprocess_mask(
            self.multitensor_system.task, x_mask, y_mask
        )
        
        return output, x_mask, y_mask


# Factory function for creating model variants
def create_arc_compressor(task, variant='advanced', **kwargs):
    """
    Factory function to create different model variants.
    
    Args:
        task: ARC task object
        variant: Model variant ('basic', 'advanced', 'competition')
        **kwargs: Additional configuration parameters
    
    Returns:
        ARCCompressor model instance
    """
    if variant == 'basic':
        from .arc_compressor_basic import ARCCompressor
        return ARCCompressor(task)
    
    elif variant == 'advanced':
        config = {
            'n_layers': 6,
            'base_dim': 32,
            'attention_heads': 8,
            'use_adaptive_arch': True,
            'use_attention': True,
            'use_progressive_refinement': True,
            'ensemble_size': 3,
        }
        config.update(kwargs)
        return AdvancedARCCompressor(task, config)
    
    elif variant == 'competition':
        # Top competition configuration
        config = {
            'n_layers': 8,
            'base_dim': 64,
            'attention_heads': 16,
            'dropout': 0.05,
            'use_adaptive_arch': True,
            'use_attention': True,
            'use_progressive_refinement': True,
            'ensemble_size': 5,
            'max_adaptive_layers': 12,
        }
        config.update(kwargs)
        return AdvancedARCCompressor(task, config)
    
    else:
        raise ValueError(f"Unknown variant: {variant}")