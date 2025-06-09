from typing import List, Optional, Tuple, Dict, Any, Set, Union
import json
import numpy as np
import torch
from arc_multitensor import ARCMultiTensorSystem

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

class ARCTask:
    """Processes ARC-AGI-2 tasks, handling grid shapes, tensor creation, and multi-tensor system setup."""
    def __init__(self, task_id: str, problem: Dict[str, List], solution: Optional[List] = None):
        self.task_id = task_id
        self.n_train = len(problem.get('train', []))
        self.n_test = len(problem.get('test', []))
        self.n_examples = self.n_train + self.n_test
        if self.n_train == 0 or self.n_test == 0:
            raise ValueError(f"Task {task_id} must have at least one train and test example.")
        self.raw_problem = problem
        self.grid_shapes = self._extract_grid_shapes(problem)
        self._infer_test_output_shapes()
        self._initialize_multitensor_system()
        self._generate_masks()
        self.problem_tensor = self._build_problem_tensor(problem)
        self.solution_tensor = self._build_solution_tensor(solution) if solution else None
        self.solution_hash = hash(tuple(tuple(map(tuple, grid)) for grid in solution)) if solution else None

    def _extract_grid_shapes(self, problem: Dict[str, List]) -> List[Tuple[List[int], Optional[List[int]]]]:
        shapes = []
        for split in ['train', 'test']:
            for example in problem[split]:
                input_shape = list(np.array(example['input']).shape)
                output_shape = list(np.array(example['output']).shape) if 'output' in example else None
                shapes.append((input_shape, output_shape))
        return shapes

    def _infer_test_output_shapes(self) -> None:
        self.input_output_same_size = all(
            tuple(inp) == tuple(out) for inp, out in self.grid_shapes[:self.n_train] if out
        )
        self.all_inputs_same_size = len({tuple(shape[0]) for shape in self.grid_shapes}) == 1
        self.all_outputs_same_size = len({tuple(shape[1]) for shape in self.grid_shapes if shape[1]}) == 1
        for i in range(self.n_train, self.n_examples):
            if self.input_output_same_size:
                self.grid_shapes[i] = (self.grid_shapes[i][0], self.grid_shapes[i][0])
            elif self.all_outputs_same_size:
                self.grid_shapes[i] = (self.grid_shapes[i][0], self.grid_shapes[0][1])
            else:
                max_x, max_y = self._find_max_dimensions()
                self.grid_shapes[i] = (self.grid_shapes[i][0], [max_x, max_y])

    def _find_max_dimensions(self) -> Tuple[int, int]:
        max_x, max_y = 0, 0
        for input_shape, output_shape in self.grid_shapes:
            if input_shape:
                max_x, max_y = max(max_x, input_shape[0]), max(max_y, input_shape[1])
            if output_shape:
                max_x, max_y = max(max_x, output_shape[0]), max(max_y, output_shape[1])
        return max_x, max_y

    def _initialize_multitensor_system(self) -> None:
        self.max_x = max(shape[i][0] for shape in self.grid_shapes for i in range(2) if shape[i])
        self.max_y = max(shape[i][1] for shape in self.grid_shapes for i in range(2) if shape[i])
        colors: Set[int] = set()
        for split in ['train', 'test']:
            for example in self.raw_problem[split]:
                for grid in [example['input'], example.get('output', [])]:
                    for row in grid:
                        colors.update(row)
        colors.add(0)
        self.colors = sorted(colors)
        self.n_colors = len(self.colors) - 1
        self.multitensor_system = ARCMultiTensorSystem(
            n_examples=self.n_examples,
            n_colors=self.n_colors,
            n_x=self.max_x,
            n_y=self.max_y,
            task=self
        )

    def _build_problem_tensor(self, problem: Dict[str, List]) -> torch.Tensor:
        tensor = np.zeros((self.n_examples, self.n_colors + 1, self.max_x, self.max_y, 2))
        for split, n_examples in [('train', self.n_train), ('test', self.n_test)]:
            for idx, example in enumerate(problem[split]):
                example_idx = idx if split == 'train' else self.n_train + idx
                for mode, mode_idx in [('input', 0), ('output', 1)]:
                    if split == 'test' and mode == 'output':
                        continue
                    grid = np.array(example.get(mode, np.zeros(self.grid_shapes[example_idx][1])))
                    grid_tensor = self._grid_to_tensor(grid)
                    min_x, min_y = min(grid_tensor.shape[1], self.max_x), min(grid_tensor.shape[2], self.max_y)
                    tensor[example_idx, :, :min_x, :min_y, mode_idx] = grid_tensor[:, :min_x, :min_y]
        return torch.argmax(torch.from_numpy(tensor), dim=1).to(torch.get_default_device())

    def _grid_to_tensor(self, grid: np.ndarray) -> np.ndarray:
        grid_tensor = np.zeros((self.n_colors + 1, *grid.shape))
        for c_idx, color in enumerate(self.colors):
            grid_tensor[c_idx] = (grid == color).astype(int)
        return grid_tensor

    def _build_solution_tensor(self, solution: List[List[List[int]]]) -> torch.Tensor:
        if len(solution) != self.n_test:
            raise ValueError(f"Expected {self.n_test} solution grids, got {len(solution)}.")
        tensor = np.zeros((self.n_test, self.n_colors + 1, self.max_x, self.max_y))
        for idx, grid in enumerate(solution):
            grid_tensor = self._grid_to_tensor(np.array(grid))
            min_x, min_y = min(grid_tensor.shape[1], self.max_x), min(grid_tensor.shape[2], self.max_y)
            tensor[idx, :, :min_x, :min_y] = grid_tensor[:, :min_x, :min_y]
        return torch.argmax(torch.from_numpy(tensor), dim=1).to(torch.get_default_device())

    def _generate_masks(self) -> None:
        self.masks = np.zeros((self.n_examples, self.max_x, self.max_y, 2))
        for idx, (input_shape, output_shape) in enumerate(self.grid_shapes):
            for mode_idx, shape in enumerate([input_shape, output_shape]):
                if shape:
                    x_mask = np.arange(self.max_x) < shape[0]
                    y_mask = np.arange(self.max_y) < shape[1]
                    self.masks[idx, :, :, mode_idx] = np.outer(x_mask, y_mask)
        self.masks = torch.from_numpy(self.masks).to(torch.get_default_dtype()).to(torch.get_default_device())

def load_and_process_tasks(split: str, task_ids: List[Union[str, int]]) -> List[ARCTask]:
    valid_splits = ['train', 'evaluation', 'test']
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}.")
    try:
        with open(f'dataset/arc-agi_{split}_challenges.json', 'r') as f:
            problems = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find dataset/arc-agi_{split}_challenges.json")
    solutions = None
    if split != 'test':
        try:
            with open(f'dataset/arc-agi_{split}_solutions.json', 'r') as f:
                solutions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find dataset/arc-agi_{split}_solutions.json")
    task_names = list(problems.keys())
    invalid_ids = []
    selected_tasks = []
    for task_id in task_ids:
        if isinstance(task_id, int):
            if task_id < 0 or task_id >= len(task_names):
                invalid_ids.append(str(task_id))
                continue
            task_name = task_names[task_id]
        else:
            if task_id not in task_names:
                invalid_ids.append(task_id)
                continue
            task_name = task_id
        selected_tasks.append(ARCTask(
            task_name,
            problems[task_name],
            solutions.get(task_name) if solutions else None
        ))
    if invalid_ids:
        raise ValueError(f"Some tasks in {invalid_ids} were not found in {split} split.")
    return selected_tasks