import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
from dataclasses import dataclass
from collections import Counter

from .arc_multitensor import multitensor_systems


@dataclass
class TaskStatistics:
    """Statistics for task complexity analysis."""
    n_train_examples: int
    n_test_examples: int
    avg_input_size: float
    avg_output_size: float
    max_grid_size: int
    color_diversity: int
    pattern_complexity: float
    transformation_type: str
    difficulty_score: float


class AdvancedDataAugmentation:
    """Advanced data augmentation for ARC tasks."""
    
    def __init__(self, augment_probability: float = 0.3):
        self.augment_probability = augment_probability
        self.augmentation_methods = [
            self._rotate_90,
            self._flip_horizontal,
            self._flip_vertical,
            self._transpose,
            self._color_permutation,
        ]
    
    def _rotate_90(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 90 degrees clockwise."""
        return np.rot90(grid, k=-1)
    
    def _flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """Flip grid horizontally."""
        return np.fliplr(grid)
    
    def _flip_vertical(self, grid: np.ndarray) -> np.ndarray:
        """Flip grid vertically."""
        return np.flipud(grid)
    
    def _transpose(self, grid: np.ndarray) -> np.ndarray:
        """Transpose grid."""
        return grid.T
    
    def _color_permutation(self, grid: np.ndarray) -> np.ndarray:
        """Apply random color permutation while preserving structure."""
        unique_colors = np.unique(grid)
        if len(unique_colors) <= 2:  # Don't permute simple cases
            return grid
        
        # Create random permutation
        perm = np.random.permutation(unique_colors)
        color_map = dict(zip(unique_colors, perm))
        
        # Apply permutation
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        
        return result
    
    def augment_example(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent augmentation to input-output pair."""
        if np.random.random() > self.augment_probability:
            return input_grid, output_grid
        
        # Choose random augmentation
        aug_method = np.random.choice(self.augmentation_methods)
        
        try:
            aug_input = aug_method(input_grid)
            aug_output = aug_method(output_grid)
            
            # Validate augmentation preserves relationships
            if self._validate_augmentation(input_grid, output_grid, aug_input, aug_output):
                return aug_input, aug_output
        except Exception:
            pass  # Fallback to original
        
        return input_grid, output_grid
    
    def _validate_augmentation(self, orig_in: np.ndarray, orig_out: np.ndarray, 
                              aug_in: np.ndarray, aug_out: np.ndarray) -> bool:
        """Validate that augmentation preserves task structure."""
        # Check that color distributions are preserved
        orig_in_colors = set(orig_in.flatten())
        aug_in_colors = set(aug_in.flatten())
        orig_out_colors = set(orig_out.flatten())
        aug_out_colors = set(aug_out.flatten())
        
        return (len(orig_in_colors) == len(aug_in_colors) and 
                len(orig_out_colors) == len(aug_out_colors))


class TaskComplexityAnalyzer:
    """Analyze task complexity for adaptive model configuration."""
    
    def __init__(self):
        self.complexity_cache = {}
    
    def analyze_task(self, task_data: Dict[str, Any]) -> TaskStatistics:
        """Comprehensive task analysis."""
        train_examples = task_data['train']
        test_examples = task_data['test']
        
        # Basic statistics
        n_train = len(train_examples)
        n_test = len(test_examples)
        
        # Size analysis
        input_sizes = []
        output_sizes = []
        max_size = 0
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            input_size = input_grid.size
            output_size = output_grid.size
            
            input_sizes.append(input_size)
            output_sizes.append(output_size)
            max_size = max(max_size, max(input_grid.shape), max(output_grid.shape))
        
        avg_input_size = np.mean(input_sizes)
        avg_output_size = np.mean(output_sizes)
        
        # Color analysis
        all_colors = set()
        for example in train_examples:
            all_colors.update(np.unique(example['input']))
            all_colors.update(np.unique(example['output']))
        
        color_diversity = len(all_colors)
        
        # Pattern complexity analysis
        pattern_complexity = self._analyze_pattern_complexity(train_examples)
        
        # Transformation type detection
        transformation_type = self._detect_transformation_type(train_examples)
        
        # Overall difficulty score
        difficulty_score = self._calculate_difficulty_score(
            n_train, avg_input_size, avg_output_size, color_diversity, pattern_complexity
        )
        
        return TaskStatistics(
            n_train_examples=n_train,
            n_test_examples=n_test,
            avg_input_size=avg_input_size,
            avg_output_size=avg_output_size,
            max_grid_size=max_size,
            color_diversity=color_diversity,
            pattern_complexity=pattern_complexity,
            transformation_type=transformation_type,
            difficulty_score=difficulty_score
        )
    
    def _analyze_pattern_complexity(self, examples: List[Dict]) -> float:
        """Analyze pattern complexity in the task."""
        complexity_scores = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Measure entropy
            input_entropy = self._calculate_entropy(input_grid)
            output_entropy = self._calculate_entropy(output_grid)
            
            # Measure spatial relationships
            spatial_complexity = self._calculate_spatial_complexity(input_grid)
            
            example_complexity = (input_entropy + output_entropy) / 2 + spatial_complexity
            complexity_scores.append(example_complexity)
        
        return np.mean(complexity_scores)
    
    def _calculate_entropy(self, grid: np.ndarray) -> float:
        """Calculate entropy of grid."""
        values, counts = np.unique(grid, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_spatial_complexity(self, grid: np.ndarray) -> float:
        """Calculate spatial complexity based on local patterns."""
        if grid.size < 4:
            return 0.0
        
        # Count unique 2x2 patterns
        patterns = set()
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                pattern = tuple(grid[i:i+2, j:j+2].flatten())
                patterns.add(pattern)
        
        max_possible_patterns = min(len(patterns), 4 ** 4)  # 4 colors, 4 positions
        complexity = len(patterns) / max(max_possible_patterns, 1)
        
        return complexity
    
    def _detect_transformation_type(self, examples: List[Dict]) -> str:
        """Detect the type of transformation in the task."""
        transformations = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            if input_grid.shape == output_grid.shape:
                if np.array_equal(input_grid, output_grid):
                    transformations.append("identity")
                elif self._is_rotation(input_grid, output_grid):
                    transformations.append("rotation")
                elif self._is_reflection(input_grid, output_grid):
                    transformations.append("reflection")
                else:
                    transformations.append("local_change")
            elif output_grid.size > input_grid.size:
                transformations.append("expansion")
            elif output_grid.size < input_grid.size:
                transformations.append("contraction")
            else:
                transformations.append("reshape")
        
        # Return most common transformation
        if transformations:
            return Counter(transformations).most_common(1)[0][0]
        return "unknown"
    
    def _is_rotation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is a rotation of input."""
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(input_grid, k), output_grid):
                return True
        return False
    
    def _is_reflection(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is a reflection of input."""
        return (np.array_equal(np.fliplr(input_grid), output_grid) or
                np.array_equal(np.flipud(input_grid), output_grid))
    
    def _calculate_difficulty_score(self, n_train: int, avg_input_size: float, 
                                  avg_output_size: float, color_diversity: int, 
                                  pattern_complexity: float) -> float:
        """Calculate overall difficulty score."""
        # Normalize components
        train_factor = min(n_train / 5.0, 1.0)  # More examples = easier
        size_factor = min((avg_input_size + avg_output_size) / 200.0, 1.0)
        color_factor = min(color_diversity / 10.0, 1.0)
        pattern_factor = min(pattern_complexity, 1.0)
        
        # Weighted combination
        difficulty = (0.2 * (1 - train_factor) +  # Fewer examples = harder
                     0.3 * size_factor +           # Larger grids = harder  
                     0.2 * color_factor +          # More colors = harder
                     0.3 * pattern_factor)         # Complex patterns = harder
        
        return min(difficulty, 1.0)


class EnhancedTask:
    """Enhanced task class with advanced preprocessing and analysis."""
    
    def __init__(self, task_name: str, problem: Dict[str, Any], solution: Optional[List] = None,
                 use_augmentation: bool = False, analyze_complexity: bool = True):
        self.task_name = task_name
        self.unprocessed_problem = problem
        self.solution = None
        
        # Initialize analyzers
        self.complexity_analyzer = TaskComplexityAnalyzer() if analyze_complexity else None
        self.augmentator = AdvancedDataAugmentation() if use_augmentation else None
        
        # Analyze task
        if self.complexity_analyzer:
            self.stats = self.complexity_analyzer.analyze_task(problem)
        else:
            self.stats = None
        
        # Process basic task properties
        self._process_basic_properties(problem)
        
        # Create enhanced multitensor system
        self._construct_enhanced_multitensor_system(problem)
        
        # Compute enhanced masks
        self._compute_enhanced_masks()
        
        # Create problem tensor with augmentation
        self._create_enhanced_problem_tensor(problem)
        
        # Process solution if provided
        if solution:
            self.solution = self._create_solution_tensor(solution)
    
    def _process_basic_properties(self, problem: Dict[str, Any]):
        """Process basic task properties."""
        self.n_train = len(problem['train'])
        self.n_test = len(problem['test'])
        self.n_examples = self.n_train + self.n_test
        
        # Collect shapes with enhanced analysis
        self.shapes = self._collect_enhanced_problem_shapes(problem)
        self._predict_enhanced_solution_shapes()
    
    def _collect_enhanced_problem_shapes(self, problem: Dict[str, Any]) -> List[List[List[int]]]:
        """Enhanced shape collection with validation."""
        shapes = []
        
        for split_name in ['train', 'test']:
            for example in problem[split_name]:
                try:
                    input_array = np.array(example['input'])
                    if input_array.ndim != 2:
                        raise ValueError(f"Input must be 2D, got {input_array.ndim}D")
                    
                    input_shape = list(input_array.shape)
                    
                    if 'output' in example:
                        output_array = np.array(example['output'])
                        if output_array.ndim != 2:
                            raise ValueError(f"Output must be 2D, got {output_array.ndim}D")
                        output_shape = list(output_array.shape)
                    else:
                        output_shape = None
                    
                    shapes.append([input_shape, output_shape])
                    
                except Exception as e:
                    warnings.warn(f"Error processing shape for {split_name}: {e}")
                    # Use fallback shape
                    shapes.append([[3, 3], [3, 3]])
        
        return shapes
    
    def _predict_enhanced_solution_shapes(self):
        """Enhanced solution shape prediction with pattern analysis."""
        # Analyze shape patterns
        input_shapes = [shape[0] for shape in self.shapes[:self.n_train]]
        output_shapes = [shape[1] for shape in self.shapes[:self.n_train] if shape[1] is not None]
        
        # Check for common patterns
        self.in_out_same_size = all(
            inp == out for inp, out in zip(input_shapes, output_shapes)
        )
        
        self.all_in_same_size = len(set(tuple(shape) for shape in input_shapes)) == 1
        self.all_out_same_size = len(set(tuple(shape) for shape in output_shapes)) == 1
        
        # Pattern-based prediction
        if self.in_out_same_size:
            # Same size pattern
            for i, shape in enumerate(self.shapes[self.n_train:], self.n_train):
                self.shapes[i][1] = shape[0].copy()
                
        elif self.all_out_same_size and output_shapes:
            # Fixed output size pattern
            reference_shape = output_shapes[0].copy()
            for i in range(self.n_train, len(self.shapes)):
                self.shapes[i][1] = reference_shape
                
        elif self.stats and self.stats.transformation_type in ['expansion', 'contraction']:
            # Size transformation pattern
            if output_shapes and input_shapes:
                size_ratios = [
                    (out[0] / inp[0], out[1] / inp[1]) 
                    for inp, out in zip(input_shapes, output_shapes)
                ]
                avg_ratio = np.mean(size_ratios, axis=0)
                
                for i in range(self.n_train, len(self.shapes)):
                    input_shape = self.shapes[i][0]
                    predicted_shape = [
                        max(1, int(input_shape[0] * avg_ratio[0])),
                        max(1, int(input_shape[1] * avg_ratio[1]))
                    ]
                    self.shapes[i][1] = predicted_shape
        else:
            # Fallback to maximum dimensions
            max_x, max_y = self._get_max_dimensions()
            for i in range(self.n_train, len(self.shapes)):
                self.shapes[i][1] = [max_x, max_y]
    
    def _get_max_dimensions(self) -> Tuple[int, int]:
        """Get maximum dimensions across all grids."""
        max_x, max_y = 3, 3  # Minimum fallback
        
        for shape_pair in self.shapes:
            for shape in shape_pair:
                if shape is not None:
                    max_x = max(max_x, shape[0])
                    max_y = max(max_y, shape[1])
        
        return max_x, max_y
    
    def _construct_enhanced_multitensor_system(self, problem: Dict[str, Any]):
        """Construct enhanced multitensor system with adaptive sizing."""
        self.n_x = max(
            shape[i][0] for shape in self.shapes 
            for i in range(2) if shape[i] is not None
        )
        self.n_y = max(
            shape[i][1] for shape in self.shapes 
            for i in range(2) if shape[i] is not None
        )
        
        # Enhanced color detection
        colors = set()
        color_frequencies = Counter()
        
        for split in ['train', 'test']:
            for example in problem[split]:
                for grid_type in ['input', 'output']:
                    if grid_type in example:
                        grid = example[grid_type]
                        for row in grid:
                            for color in row:
                                colors.add(color)
                                color_frequencies[color] += 1
        
        # Always include background color
        colors.add(0)
        
        # Sort colors by frequency for better representation
        self.colors = sorted(colors, key=lambda c: color_frequencies[c], reverse=True)
        self.n_colors = len(self.colors) - 1  # Exclude background
        
        # Create enhanced multitensor system
        self.multitensor_system = multitensor_systems.MultiTensorSystem(
            self.n_examples, self.n_colors, self.n_x, self.n_y, self
        )
    
    def _compute_enhanced_masks(self):
        """Compute enhanced masks with adaptive boundaries."""
        self.masks = np.zeros((self.n_examples, self.n_x, self.n_y, 2))
        
        for example_num, (input_shape, output_shape) in enumerate(self.shapes):
            for mode_num, shape in enumerate([input_shape, output_shape]):
                if shape is not None:
                    # Create soft boundaries for better gradient flow
                    x_mask = np.zeros(self.n_x)
                    y_mask = np.zeros(self.n_y)
                    
                    x_mask[:shape[0]] = 1.0
                    y_mask[:shape[1]] = 1.0
                    
                    # Add soft transition at boundaries
                    if shape[0] < self.n_x:
                        x_mask[shape[0]:min(shape[0] + 2, self.n_x)] = 0.5
                    if shape[1] < self.n_y:
                        y_mask[shape[1]:min(shape[1] + 2, self.n_y)] = 0.5
                    
                    # Create 2D mask
                    mask_2d = np.outer(x_mask, y_mask)
                    self.masks[example_num, :, :, mode_num] = mask_2d
        
        # Convert to tensor
        device = torch.get_default_device()
        self.masks = torch.from_numpy(self.masks).to(torch.get_default_dtype()).to(device)
    
    def _create_enhanced_problem_tensor(self, problem: Dict[str, Any]):
        """Create enhanced problem tensor with augmentation."""
        self.problem = np.zeros((self.n_examples, self.n_colors + 1, self.n_x, self.n_y, 2))
        
        example_idx = 0
        
        for split_name in ['train', 'test']:
            examples = problem[split_name]
            
            for example in examples:
                for mode_name in ['input', 'output']:
                    if mode_name not in example:
                        continue
                    
                    mode_idx = 0 if mode_name == 'input' else 1
                    
                    # Skip test outputs
                    if split_name == 'test' and mode_name == 'output':
                        continue
                    
                    # Get grid
                    grid = np.array(example[mode_name])
                    
                    # Apply augmentation for training data
                    if (self.augmentator and split_name == 'train' and 
                        mode_name == 'input' and 'output' in example):
                        
                        output_grid = np.array(example['output'])
                        aug_input, aug_output = self.augmentator.augment_example(grid, output_grid)
                        
                        # Use augmented grids
                        grid = aug_input
                        # Store augmented output for later use
                        example['_aug_output'] = aug_output
                    
                    # Convert to one-hot representation
                    grid_tensor = self._create_enhanced_grid_tensor(grid)
                    
                    # Place in problem tensor
                    h, w = grid.shape
                    self.problem[example_idx, :, :h, :w, mode_idx] = grid_tensor[:, :h, :w]
                
                example_idx += 1
        
        # Convert to tensor and apply label smoothing for training stability
        device = torch.get_default_device()
        self.problem = torch.from_numpy(self.problem).to(device)
        
        # Apply label smoothing to training examples
        if self.n_train > 0:
            smoothing = 0.01
            self.problem[:self.n_train] = (
                self.problem[:self.n_train] * (1 - smoothing) + 
                smoothing / (self.n_colors + 1)
            )
        
        # Convert to class indices for final representation
        self.problem = torch.argmax(self.problem, dim=1)
    
    def _create_enhanced_grid_tensor(self, grid: np.ndarray) -> np.ndarray:
        """Create enhanced grid tensor with error handling."""
        try:
            # Handle invalid colors
            valid_grid = np.clip(grid, 0, len(self.colors) - 1)
            
            # Create one-hot representation
            tensor = np.zeros((len(self.colors), grid.shape[0], grid.shape[1]))
            
            for color_idx, color in enumerate(self.colors):
                tensor[color_idx] = (valid_grid == color).astype(np.float32)
            
            return tensor
            
        except Exception as e:
            warnings.warn(f"Error creating grid tensor: {e}")
            # Return default tensor
            tensor = np.zeros((len(self.colors), grid.shape[0], grid.shape[1]))
            tensor[0] = 1.0  # All background
            return tensor
    
    def _create_solution_tensor(self, solution: List) -> torch.Tensor:
        """Create enhanced solution tensor with validation."""
        try:
            solution_tensor = np.zeros((self.n_test, self.n_colors + 1, self.n_x, self.n_y))
            solution_hash_data = []
            
            for example_num, grid in enumerate(solution):
                if example_num >= self.n_test:
                    break
                
                # Validate solution grid
                grid_array = np.array(grid)
                if grid_array.ndim != 2:
                    warnings.warn(f"Solution {example_num} is not 2D, skipping")
                    continue
                
                # Store for hashing
                solution_hash_data.append(tuple(map(tuple, grid)))
                
                # Convert to tensor representation
                grid_tensor = self._create_enhanced_grid_tensor(grid_array)
                
                # Clip to fit tensor dimensions
                h, w = min(grid_array.shape[0], self.n_x), min(grid_array.shape[1], self.n_y)
                solution_tensor[example_num, :, :h, :w] = grid_tensor[:, :h, :w]
            
            # Create solution hash
            self.solution_hash = hash(tuple(solution_hash_data))
            
            # Convert to class indices
            device = torch.get_default_device()
            solution_tensor = torch.from_numpy(solution_tensor).to(device)
            return torch.argmax(solution_tensor, dim=1)
            
        except Exception as e:
            warnings.warn(f"Error creating solution tensor: {e}")
            self.solution_hash = 0
            device = torch.get_default_device()
            return torch.zeros((self.n_test, self.n_x, self.n_y), device=device)
    
    def get_adaptive_config(self) -> Dict[str, Any]:
        """Get adaptive configuration based on task analysis."""
        if not self.stats:
            return {}
        
        config = {}
        
        # Model size adaptation
        if self.stats.difficulty_score > 0.7:
            config['model_variant'] = 'competition'
            config['base_dim'] = 64
            config['n_layers'] = 8
        elif self.stats.difficulty_score > 0.4:
            config['model_variant'] = 'advanced'
            config['base_dim'] = 48
            config['n_layers'] = 6
        else:
            config['model_variant'] = 'basic'
            config['base_dim'] = 32
            config['n_layers'] = 4
        
        # Training adaptation
        if self.stats.pattern_complexity > 0.6:
            config['n_iterations'] = 3000
            config['learning_rate'] = 0.005
        else:
            config['n_iterations'] = 2000
            config['learning_rate'] = 0.01
        
        # Architecture adaptation
        config['use_attention'] = self.stats.avg_input_size > 50
        config['use_progressive_refinement'] = self.stats.max_grid_size > 10
        config['ensemble_size'] = min(5, max(3, int(self.stats.difficulty_score * 5)))
        
        return config


# Backward compatibility
class Task(EnhancedTask):
    """Backward compatible Task class."""
    
    def __init__(self, task_name: str, problem: Dict[str, Any], solution: Optional[List] = None):
        super().__init__(task_name, problem, solution, use_augmentation=False, analyze_complexity=True)


def preprocess_tasks(split: str, task_nums_or_task_names: Union[List[int], List[str]], 
                    use_augmentation: bool = False, analyze_complexity: bool = True) -> List[EnhancedTask]:
    """
    Enhanced task preprocessing with advanced features.
    
    Args:
        split: Data split ('training', 'evaluation', 'test')
        task_nums_or_task_names: Task numbers or names to process
        use_augmentation: Whether to apply data augmentation
        analyze_complexity: Whether to analyze task complexity
    
    Returns:
        List of processed EnhancedTask objects
    """
    # Load data files
    try:
        data_path = Path('data/raw')
        
        problems_file = data_path / f'arc-agi_{split}_challenges.json'
        with open(problems_file, 'r') as f:
            problems = json.load(f)
        
        solutions = None
        if split != "test":
            solutions_file = data_path / f'arc-agi_{split}_solutions.json'
            try:
                with open(solutions_file, 'r') as f:
                    solutions = json.load(f)
            except FileNotFoundError:
                warnings.warn(f"Solutions file not found: {solutions_file}")
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data files not found. Please run 'make data' first. Error: {e}")
    
    # Get task names
    task_names = list(problems.keys())
    
    # Filter tasks
    if isinstance(task_nums_or_task_names[0], int):
        # Task numbers provided
        filtered_names = [task_names[i] for i in task_nums_or_task_names if i < len(task_names)]
    else:
        # Task names provided
        filtered_names = [name for name in task_nums_or_task_names if name in task_names]
    
    # Process tasks
    processed_tasks = []
    
    for task_name in filtered_names:
        try:
            task_solution = solutions.get(task_name) if solutions else None
            
            task = EnhancedTask(
                task_name=task_name,
                problem=problems[task_name],
                solution=task_solution,
                use_augmentation=use_augmentation,
                analyze_complexity=analyze_complexity
            )
            
            processed_tasks.append(task)
            
        except Exception as e:
            warnings.warn(f"Failed to process task {task_name}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_tasks)}/{len(filtered_names)} tasks")
    
    return processed_tasks


def analyze_dataset_statistics(tasks: List[EnhancedTask]) -> Dict[str, Any]:
    """Analyze statistics across multiple tasks."""
    if not tasks or not tasks[0].stats:
        return {}
    
    stats = {
        'n_tasks': len(tasks),
        'avg_difficulty': np.mean([t.stats.difficulty_score for t in tasks]),
        'avg_complexity': np.mean([t.stats.pattern_complexity for t in tasks]),
        'transformation_types': Counter([t.stats.transformation_type for t in tasks]),
        'color_diversity': np.mean([t.stats.color_diversity for t in tasks]),
        'avg_grid_size': np.mean([t.stats.max_grid_size for t in tasks]),
    }
    
    return stats