from typing import List, Tuple, Callable, Optional, Union
import itertools
import math
import numpy as np
import torch
from torch import nn
from src.arc_multitensor import ARCMultiTensorSystem, ARCMultiTensor, apply_multitensor
from src.arc_task_processor import ARCTask

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

@apply_multitensor
def normalize(x: torch.Tensor, debias: bool = True) -> torch.Tensor:
    """Normalize a tensor to unit variance along all but the last dimension."""
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}.")
    if x.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    axes = list(range(x.ndim - 1))
    if debias:
        x = x - torch.mean(x, dim=axes, keepdim=True)
    variance = torch.mean(x**2, dim=axes, keepdim=True)
    return x / torch.sqrt(variance + 1e-8)

@apply_multitensor
def affine(x: torch.Tensor, weights: Union[torch.Tensor, List[torch.Tensor]], use_bias: bool = False) -> torch.Tensor:
    """Apply a linear transformation along the channel dimension."""
    if isinstance(x, ARCMultiTensor):
        x = x[[1] * 5]
    if isinstance(weights, ARCMultiTensor):
        weights = weights[[1] * 5]
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}.")
    if isinstance(weights, torch.Tensor):
        weights = [weights]
    if not weights or not isinstance(weights[0], torch.Tensor):
        raise ValueError("Weights must include a valid weight matrix.")
    if weights[0].ndim != 2 or weights[0].shape[0] != x.shape[-1]:
        raise ValueError(f"Weight matrix shape {weights[0].shape} incompatible with input shape {x.shape}.")
    x = torch.matmul(x, weights[0])
    if use_bias:
        if len(weights) < 2 or not isinstance(weights[1], torch.Tensor):
            raise ValueError("Bias must be provided when use_bias is True.")
        if weights[1].shape[-1] != weights[0].shape[-1]:
            raise ValueError(f"Bias shape {weights[1].shape} incompatible with weight matrix {weights[0].shape}.")
        x = x + weights[1]
    return x

def add_residual(layer: Callable) -> Callable:
    def residual_layer(
        x: torch.Tensor,
        projection_weights: List[List[torch.Tensor]],
        *args,
        use_bias: bool = False,
        pre_norm: bool = False,
        post_norm: bool = False,
        **kwargs
    ) -> torch.Tensor:
        if isinstance(projection_weights, ARCMultiTensor):
            projection_weights = projection_weights[[1] * 5]
        if not isinstance(projection_weights, list) or len(projection_weights) != 2:
            raise ValueError("projection_weights must contain two lists of weights.")
        z = normalize(x) if pre_norm else x
        if isinstance(projection_weights[0], list):
            z = torch.matmul(z, projection_weights[0][0])
            if use_bias:
                z = z + projection_weights[0][1]
        z = layer(z, projection_weights=projection_weights, *args, **kwargs)
        if post_norm:
            z = normalize(z)
        if z.shape[-1] != x.shape[-1]:
            if projection_weights[1][0].shape[1] != z.shape[-1]:
                raise ValueError(f"Second projection weight shape {projection_weights[1][0].shape} incompatible with layer output {z.shape}.")
        if isinstance(projection_weights[1], list):
            z = torch.matmul(z, projection_weights[1][0])
            if use_bias:
                z = z + projection_weights[1][1]
        if z.shape != x.shape:
            raise ValueError(f"Residual output shape {z.shape} does not match input shape {x.shape}.")
        return x + z
    return residual_layer

def channel_layer(target_capacity: torch.Tensor, posterior: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(target_capacity, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor for target_capacity, got {type(target_capacity)}.")
    mean, local_capacity_adjustment = posterior
    if not isinstance(mean, torch.Tensor) or not isinstance(local_capacity_adjustment, torch.Tensor):
        raise ValueError("Posterior must contain two tensors.")
    if mean.shape[-1] != local_capacity_adjustment.shape[-1]:
        raise ValueError(f"Mean shape {mean.shape} and adjustment shape {local_capacity_adjustment.shape} mismatch.")
    axes = tuple(range(len(mean.shape) - 1))
    dimensionality = np.prod(mean.shape[:-1]) or 1

    min_capacity = torch.tensor(0.5, device=mean.device)
    init_capacity = torch.tensor(10000.0, device=mean.device)
    target_capacity = 10 * target_capacity

    global_capacity = torch.exp(target_capacity) * init_capacity + min_capacity
    output_scaling = 1 - torch.exp(-global_capacity / dimensionality * 2)

    local_weight = (
        target_capacity + local_capacity_adjustment - torch.mean(local_capacity_adjustment, dim=axes, keepdim=True)
    )
    local_capacity = torch.exp(local_weight) * init_capacity + min_capacity

    noise_std = torch.exp(-local_capacity / dimensionality)
    noise_var = noise_std**2
    signal_var = 1 - noise_var
    signal_std = torch.sqrt(torch.clamp(signal_var, 0.0, 1.0))

    normalized_mean = normalize(mean, debias=True)
    z = signal_std * normalized_mean + noise_std * torch.randn_like(normalized_mean)
    z = output_scaling * z

    kl = 0.5 * (noise_var + signal_var * normalized_mean**2 - 1) + local_capacity / dimensionality
    return z, kl

def decode_latents(
    target_capacities: ARCMultiTensor,
    decode_weights: ARCMultiTensor,
    multiposteriors: ARCMultiTensor
) -> Tuple[ARCMultiTensor, List[torch.Tensor], List[str]]:
    kl_divergences = []
    kl_names = []

    @apply_multitensor
    def decode(
        target_capacity: torch.Tensor,
        decode_weight: List[torch.Tensor],
        posterior: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        mean, adjustment = posterior
        z, kl = channel_layer(target_capacity, (mean, adjustment))
        kl_divergences.append(kl)
        kl_names.append(str(target_capacities.system.get_shape([1] * 5)))
        if isinstance(decode_weight, list):
            decode_weight = decode_weight[0]
        return torch.matmul(z, decode_weight)

    decoded = decode(target_capacities, decode_weights, multiposteriors)
    if not isinstance(decoded, ARCMultiTensor):
        raise TypeError(f"Expected ARCMultiTensor from decode, got {type(decoded)}.")
    return decoded, kl_divergences, kl_names

def share_direction(
    residual: ARCMultiTensor,
    share_weights: ARCMultiTensor,
    direction: int,
) -> ARCMultiTensor:
    if direction not in (1, -1):
        raise ValueError("Direction must be 1 (up) or -1 (down).")

    system = residual.system
    down_weights = apply_multitensor(lambda w: w[0] if isinstance(w, list) else w)(share_weights)
    up_weights = apply_multitensor(lambda w: w[1] if isinstance(w, list) else w)(share_weights)
    
    @apply_multitensor
    def apply_affine(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if isinstance(w, list):
            w = w[0]
        if isinstance(w, ARCMultiTensor):
            w = w[[1] * 5]
        return torch.matmul(x, w)

    x = apply_affine(residual, down_weights)

    @apply_multitensor
    def communicate(x: torch.Tensor) -> torch.Tensor:
        tensors = []
        base_shape = x.shape
        for dims in system:
            for other_dims in system:
                if direction == 1:  # Up: include lower or equal dimensions
                    if all(o <= d for o, d in zip(other_dims, dims)):
                        t = x
                        # Adjust shape to match base_shape
                        current_shape = list(t.shape)
                        target_shape = list(base_shape)
                        # Pad dimensions that are missing
                        for i in range(len(current_shape) - 1):
                            while len(t.shape) <= i:
                                t = t.unsqueeze(i)
                        # Unsqueeze dimensions where other_dims < dims
                        for i, (o, d) in enumerate(zip(other_dims, dims)):
                            if o < d and t.shape[i] == 1:
                                t = t.expand(-1 if j != i else target_shape[i] for j in range(len(t.shape)))
                        if t.shape[-1] != target_shape[-1]:
                            continue
                        tensors.append(t)
                else:  # Down: include higher or equal dimensions
                    if all(o >= d for o, d in zip(other_dims, dims)):
                        t = x
                        for i, (o, d) in enumerate(zip(other_dims, dims)):
                            if o > d:
                                axis = i
                                if (system.task.input_output_same_size or system.task.all_outputs_same_size) and i in (3, 4):
                                    masks = system.task.masks
                                    masks = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
                                    for _ in range(sum(other_dims[1:3])):
                                        masks = masks[:, None, ...]
                                    if i == 3 and dims[3] == 0:
                                        masks = masks[..., 0]
                                    elif i == 4 and dims[4] == 0:
                                        masks = masks[..., 0, :]
                                    masks = masks[..., None]
                                    t = torch.sum(t * masks, dim=axis) / (torch.sum(masks, dim=axis) + 1e-4)
                                else:
                                    t = torch.mean(t, dim=axis)
                        if t.shape[-1] != base_shape[-1]:
                            continue
                        # Ensure broadcastable shape
                        while len(t.shape) < len(base_shape):
                            t = t.unsqueeze(0)
                        tensors.append(t)
        if not tensors:
            return torch.zeros(base_shape, device=x.device)
        # Ensure all tensors have the same shape
        tensors = [t.expand(base_shape) if t.shape != base_shape else t for t in tensors]
        return sum(tensors)

    x = communicate(x)
    x = normalize(x)
    x = apply_affine(x, up_weights)
    result = apply_multitensor(lambda x, y: x + y)(residual, x)
    if not isinstance(result, ARCMultiTensor):
        raise TypeError(f"Expected ARCMultiTensor from share_direction, got {type(result)}.")
    return result

def share_up(residual: ARCMultiTensor, share_weights: ARCMultiTensor) -> ARCMultiTensor:
    return share_direction(residual, share_weights, 1)

def share_down(residual: ARCMultiTensor, share_weights: ARCMultiTensor) -> ARCMultiTensor:
    return share_direction(residual, share_weights, -1)

def only_do_for_certain_shapes(*shapes: Tuple[int, ...]) -> Callable:
    shapes_set = set(shapes)
    def decorator(fn: Callable):
        @apply_multitensor
        def filtered_fn(x: Union[torch.Tensor, ARCMultiTensor], *args, **kwargs) -> Union[torch.Tensor, ARCMultiTensor]:
            if isinstance(x, ARCMultiTensor):
                dims = x.system.get_shape([1] * 5)
                if tuple(dims) in shapes_set:
                    return fn(x, *args, **kwargs)
                return x
            return x
        return filtered_fn
    return decorator

@apply_multitensor
@add_residual
def softmax(x: torch.Tensor, projection_weights: List[List[torch.Tensor]], *args, **kwargs) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}.")
    if x.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    input_channels = x.shape[-1]
    axes = list(range(1, x.ndim - 1))  # Exclude batch dimension
    softmax_outputs = []
    for dim in axes:
        offsets = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - offsets)
        softmax_outputs.append(exp_x / torch.sum(exp_x, dim=dim, keepdim=True))
    output = torch.cat(softmax_outputs, dim=-1)
    return output

def make_directional_layer(fn: Callable, diagonal_fn: Callable) -> Callable:
    @apply_multitensor
    def directional_layer(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(x)}.")
        if x.shape[:-1] != masks.shape[:3]:
            raise ValueError(f"Input shape {x.shape} does not match mask shape {masks.shape}.")
        system = x.system if isinstance(x, ARCMultiTensor) else None
        dims = system.get_shape([1] * 5) if system else [1] * 5
        direction_dim = sum(dims[:2]) + 1  # Adjust for direction dimension
        channels_dim = x.ndim - 1

        masks = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
        if dims[4] == 0:
            masks = masks[..., 0]
        if dims[3] == 0:
            masks = masks[..., 0]
        for _ in range(sum(dims[1:3])):
            masks = masks[:, None, ...]
        masks = masks[..., None]
        x = x * masks

        n_directions = dims[2]
        zero_tensor = torch.zeros_like(x)

        results = []
        for channel_split in range(2):  # Forward, backward
            direction_results = []
            for direction_split in range(2):  # Forward, backward
                for d in range(8):  # x, x+y, y, y-x, -x, -x-y, -y, -x+y
                    if d % 2 == 0:  # Cardinal directions
                        cardinal_idx = d // 2
                        direction_dim_idx = 1 + cardinal_idx  # x or y dimension
                        if dims[3 + cardinal_idx] > 0:
                            x_slice = x[..., channel_split::2]
                            mask_slice = masks[..., 0]
                            if direction_split + channel_split == 1:
                                x_slice = torch.flip(x_slice, dims=[direction_dim_idx])
                                mask_slice = torch.flip(mask_slice, dims=[direction_dim_idx])
                            result = fn(x_slice, direction_dim_idx, mask_slice)
                            if direction_split + channel_split == 1:
                                result = torch.flip(result, dims=[direction_dim_idx])
                        else:
                            result = zero_tensor
                    else:  # Diagonal directions
                        if dims[3] == 1 and dims[4] == 1:
                            diagonal_idx = d // 2
                            x_slice = x[..., channel_split::2]
                            mask_slice = masks[..., 0]
                            flip_dims = []
                            if (direction_split + channel_split + diagonal_idx) % 2 == 1:
                                flip_dims.append(direction_dim)
                            if direction_split + channel_split == 1:
                                flip_dims.append(direction_dim + 1)
                            if flip_dims:
                                x_slice = torch.flip(x_slice, dims=flip_dims)
                                mask_slice = torch.flip(mask_slice, dims=flip_dims)
                            result = diagonal_fn(x_slice, direction_dim, direction_dim + 1, mask_slice)
                            if flip_dims:
                                result = torch.flip(result, dims=flip_dims)
                        else:
                            result = zero_tensor
                    direction_results.append(result)
                direction_results = torch.stack(direction_results, dim=channels_dim)
            results.append(direction_results)
        output = torch.cat(results, dim=-1)
        return output
    return directional_layer

def cummax_(x: torch.Tensor, dim: int, masks: torch.Tensor) -> torch.Tensor:
    mask_penalty = 1e3 * (1 - masks)
    max_vals = torch.max(x - mask_penalty, dim=dim, keepdim=True)[0]
    min_vals = torch.min(x + mask_penalty, dim=dim, keepdim=True)[0]
    x = torch.cummax(x - mask_penalty, dim=dim)[0]
    x = x + mask_penalty + 1e-3
    return (x - min_vals) / (max_vals - min_vals + 1e-5) * 2 - 1

def diagonal_cummax_(x: torch.Tensor, dim1: int, dim2: int, masks: torch.Tensor) -> torch.Tensor:
    mask_penalty = 1e3 * (1 - masks)
    min_dim = min(x.shape[dim1], x.shape[dim2])
    n_iters = max(int(math.ceil(math.log2(min_dim))), 1)
    
    max_x = x - mask_penalty
    for sign in [1, -1]:
        for i in range(n_iters):
            shift = sign * (2 ** i)
            shifted_x = diagonal_shift(max_x, dim1, dim2, shift, -1e3)
            max_x = torch.max(max_x, shifted_x)
    cummax_x = max_x + mask_penalty

    min_x = x + mask_penalty
    for sign in [1, -1]:
        for i in range(n_iters):
            shift = sign * (2 ** i)
            shifted_x = diagonal_shift(min_x, dim1, dim2, shift, 1e3)
            min_x = torch.min(min_x, shifted_x)
    min_x -= mask_penalty

    return ((cummax_x - min_x) / (max_x - min_x + 1e-5) * 2 - 1) * masks

def diagonal_shift(x: torch.Tensor, dim1: int, dim2: int, shift_amount: int, pad_value: float) -> torch.Tensor:
    result = x.clone()
    for dim in (dim1, dim2):
        padding = pad_value + torch.zeros_like(result.narrow(dim, 0, abs(shift_amount)))
        if shift_amount >= 0:
            narrowed = result.narrow(dim, 0, result.shape[dim] - abs(shift_amount))
            result = torch.cat([padding, narrowed], dim=dim)
        else:
            narrowed = result.narrow(dim, abs(shift_amount), result.shape[dim] - abs(shift_amount))
            result = torch.cat([narrowed, padding], dim=dim)
    return result

cummax = apply_multitensor(
    only_do_for_certain_shapes((1, 1, 1, 1, 1), (1, 0, 1, 1, 1))(
        add_residual(
            make_directional_layer(cummax_, diagonal_cummax_)
        )
    )
)

def shift_(x: torch.Tensor, dim: int, masks: torch.Tensor) -> torch.Tensor:
    padding = torch.zeros_like(x.narrow(dim, 0, 1))
    narrowed = x.narrow(dim, 0, x.shape[dim] - 1)
    return torch.cat([padding, narrowed], dim=dim)

def diagonal_shift_(x: torch.Tensor, dim1: int, dim2: int, masks: torch.Tensor, shift_amount: int = 1, pad_value: float = 0) -> torch.Tensor:
    return diagonal_shift(x, dim1, dim2, shift_amount, pad_value)

shift = apply_multitensor(
    only_do_for_certain_shapes((1, 1, 1, 1, 1), (1, 0, 1, 1, 1))(
        add_residual(
            make_directional_layer(shift_, diagonal_shift_)
        )
    )
)

directional_dims = [(i, j, k, l, m) for i, j, k, l, m in itertools.product(range(2), repeat=5) if k == 1]

@apply_multitensor
@only_do_for_certain_shapes(*directional_dims)
def direction_share(
    x: torch.Tensor,
    weights: List[List[List[torch.Tensor]]],
    pre_norm: bool = True,
    use_bias: bool = False
) -> torch.Tensor:
    if isinstance(x, ARCMultiTensor):
        x = x[[1] * 5]
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}.")
    if len(weights) != 8 or any(len(w) != 8 for w in weights[0]):
        raise ValueError("Weights must be a list of 8x8 weight matrices.")
    z = normalize(x) if pre_norm else x
    system = x.system if isinstance(x, ARCMultiTensor) else None
    dims = system.get_shape([1] * 5) if system else [1] * 5
    direction_dim = sum(dims[:2]) + 2  # Adjust for direction dimension
    if x.shape[direction_dim] != 8:
        raise ValueError(f"Expected 8 directions in dimension {direction_dim}, got {x.shape[direction_dim]}")
    x_slices = torch.unbind(z, dim=direction_dim)
    z_slices = torch.unbind(z, dim=direction_dim)
    coefficients = [1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2]

    outputs = list(x_slices)
    for d1 in range(8):
        for d2 in range(8):
            c = coefficients[(d2 - d1) % 8]
            outputs[d1] = outputs[d1] + c * affine(z_slices[d2], weights[0][d1][d2], use_bias=use_bias)

    result = torch.stack(outputs, dim=direction_dim)
    if system:
        return ARCMultiTensor(system, result)
    return result

@apply_multitensor
@add_residual
def nonlinear(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}.")
    return nn.functional.silu(x)

def postprocess_mask(task: ARCTask, x_mask: torch.Tensor, y_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_mask.shape[:2] != (task.n_examples, task.max_x) or y_mask.shape[:2] != (task.n_examples, task.max_y):
        raise ValueError("Mask shapes do not match task dimensions.")

    x_modifier = np.zeros((task.n_examples, task.max_x, 2))
    y_modifier = np.zeros((task.n_examples, task.max_y, 2))
    for i in range(task.n_examples):
        max_x = max(task.grid_shapes[i][0][0], task.grid_shapes[i][1][0] if task.grid_shapes[i][1] else 0)
        max_y = max(task.grid_shapes[i][0][1], task.grid_shapes[i][1][1] if task.grid_shapes[i][1] else 0)
        x_modifier[i, max_x:, :] = -1000
        y_modifier[i, max_y:, :] = -1000

    x_mask = x_mask + torch.from_numpy(x_modifier).to(x_mask.device, x_mask.dtype)
    y_mask = y_mask + torch.from_numpy(y_modifier).to(y_mask.device, y_mask.dtype)
    return x_mask, y_mask