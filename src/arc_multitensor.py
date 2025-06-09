from typing import List, Optional, Any, Union, Callable
import numpy as np
import torch

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

class ARCMultiTensorSystem:
    """Manages multi-dimensional configurations for ARC-AGI tasks, handling examples, colors, directions, and (x, y) positions."""
    NUM_DIMENSIONS: int = 5  # Fixed dimensions: examples, colors, directions, x, y

    def __init__(self, n_examples: int, n_colors: int, n_x: int, n_y: int, task: Any):
        if any(dim <= 0 for dim in [n_examples, n_colors, n_x, n_y]):
            raise ValueError("All dimensions (n_examples, n_colors, n_x, n_y) must be positive.")
        if task is None:
            raise ValueError("Task cannot be None.")

        self.n_examples = n_examples
        self.n_colors = n_colors
        self.n_directions = 8  # Fixed number of directions
        self.n_x = n_x
        self.n_y = n_y
        self.task = task
        self.dim_lengths = [self.n_examples, self.n_colors, self.n_directions, self.n_x, self.n_y]

    def is_valid_combination(self, dims: List[int]) -> bool:
        if len(dims) != self.NUM_DIMENSIONS:
            raise ValueError(f"Expected {self.NUM_DIMENSIONS} dimensions, got {len(dims)}.")
        if (dims[3] or dims[4]) and not dims[0]:  # x or y requires examples
            return False
        if sum(dims[1:]) == 0:  # At least one of colors, directions, x, y needed
            return False
        return True

    def get_shape(self, dims: List[int], extra_dim: Optional[int] = None) -> List[int]:
        shape = [length for i, length in enumerate(self.dim_lengths) if dims[i]]
        if extra_dim is not None:
            shape.append(extra_dim)
        return shape

    def _generate_combinations(self):
        for i in range(1 << self.NUM_DIMENSIONS):
            dims = [(i >> bit) & 1 for bit in range(self.NUM_DIMENSIONS)]
            if self.is_valid_combination(dims):
                yield dims

    def __iter__(self):
        return self._generate_combinations()

    def create_multitensor(self, default: Any = None) -> 'ARCMultiTensor':
        def build_nested(depth: int) -> Union[List, Any]:
            if depth == self.NUM_DIMENSIONS:
                return default
            return [build_nested(depth + 1) for _ in range(2)]
        return ARCMultiTensor(build_nested(0), self)

class ARCMultiTensor:
    def __init__(self, data: List, system: ARCMultiTensorSystem):
        self.data = data
        self.system = system

    def __getitem__(self, dims: List[int]) -> Any:
        if len(dims) != self.system.NUM_DIMENSIONS:
            raise ValueError(f"Expected {self.system.NUM_DIMENSIONS} dimensions, got {len(dims)}.")
        node = self.data
        for dim in dims:
            node = node[dim]
        return node

    def __setitem__(self, dims: List[int], value: Any) -> None:
        if len(dims) != self.system.NUM_DIMENSIONS:
            raise ValueError(f"Expected {self.NUM_DIMENSIONS} dimensions, got {len(dims)}.")
        node = self.data
        for dim in dims[:-1]:
            node = node[dim]
        node[dims[-1]] = value

def apply_multitensor(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        multitensor_system = None
        is_multi_mode = False
        # Check args for ARCMultiTensor
        for arg in args:
            if isinstance(arg, ARCMultiTensor):
                is_multi_mode = True
                multitensor_system = arg.system
                break
        # Check kwargs for ARCMultiTensor
        if not is_multi_mode:
            for value in kwargs.values():
                if isinstance(value, ARCMultiTensor):
                    is_multi_mode = True
                    multitensor_system = value.system
                    break
        # Non-multi mode: call function directly
        if not is_multi_mode:
            return fn(*args, **kwargs)
        # Multi mode: apply function for each dimension combination
        result = multitensor_system.create_multitensor()
        for dims in multitensor_system:
            new_args = [arg[dims] if isinstance(arg, ARCMultiTensor) else arg for arg in args]
            new_kwargs = {k: v[dims] if isinstance(v, ARCMultiTensor) else v for k, v in kwargs.items()}
            result[dims] = fn(*new_args, **new_kwargs)
        return result
    return wrapper