# ARC-AGI-2 Solver
[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![pytest](https://github.com/sudo-de/arc-agi-2-solver/actions/workflows/pytest.yml/badge.svg)](https://github.com/sudo-de/arc-agi-2-solver/actions)

## Overview
Welcome to the ARC-AGI-2 Solver! This project implements a sophisticated neural network model designed to solve the Abstraction and Reasoning Challenge (ARC-AGI-2). Our model leverages a Variational Autoencoder (VAE) decoder architecture with custom layers tailored for spatial reasoning and pattern recognition, making it a powerful tool for tackling complex ARC tasks.

- **[Abstraction and Reasoning Challenge (ARC-AGI-2)](https://www.kaggle.com/competitions/arc-prize-2025)**

## Key Features

- **Multi-tensor System**: Handles different dimensional configurations for flexible reasoning
- **Custom Neural Layers**: Specialized layers for directional operations (cummax, shift, softmax)
- **Modular Architecture**: Organized into distinct modules for task processing (`arc_task_processor`), multi-tensor operations (`arc_multitensor`), model training (`arc_trainer`), and visualization (`arc_visualizer`).
- **VAE-Based Model**: Implements `ARCVAE` for encoding and decoding ARC grids, with support for multi-dimensional tensor systems.
- **Robust Testing**: Comprehensive pytest suite covering task processing, model training, and solution selection, ensuring reliability.
- **Visualization Tools**: Generates high-quality plots for task inputs, outputs, and predicted solutions using Matplotlib.
- **Symmetry Preservation**: Maintains x-y symmetry in weight initialization.
- **Parallel Processing**: Multi-GPU support for solving multiple tasks simultaneously.

## Architecture Components

### Core Modules

1. **ARCCompressor**: Main VAE decoder model
2. **MultiTensorSystem**: Handles multi-dimensional tensor operations
3. **Custom Layers**: Specialized neural network layers
4. **Preprocessing**: Task data handling and tensor creation
5. **Solution Selection**: Solution tracking and optimization

### Model Layers

- **Decoding Layer**: VAE latent variable decoding
- **Share Layers**: Multi-tensor communication (up/down)
- **Softmax Layer**: Attention over dimensional combinations
- **Directional Layers**: Cumulative maximum and shift operations
- **Direction Share**: Directional communication layer
- **Nonlinear Layer**: SiLU activation with residual connections

## Installation

1. **Clone the Repository**:
   
```bash
# Clone the repository
git clone https://github.com/example/arc-agi-solver.git
cd arc-agi-2-solver

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

### Quick Start

1. **Set Up a Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On MacOS
venv\Scripts\activate # On Windows
```
2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
3. **Prepare Data:**
- Place ARC-AGI-2 task files (e.g., arc-agi_test_challenges.json) in the data/ directory.
- Update src/arc_task_processor.py to point to your data path if necessary.
4. **Running Tests**
```bash
pytest tests/ -v --cov=src
```

### Quick Start

```python
from src.solve_task import solve_multiple_tasks, create_kaggle_submission

# Solve ARC tasks
results = solve_multiple_tasks(
    task_names=['task_001', 'task_002'],
    split='test',
    time_limit_per_task=300.0,
    n_train_iterations=2000
)

# Create submission file
create_kaggle_submission(results, 'submission.json')
```

### Command Line Interface

```bash
# Solve all tasks in test split
python -m src.solve_task --split test --output submission.json

# Solve with custom parameters
python -m src.solve_task --split test --time-limit 180 --iterations 1500 --workers 4
```

### Kaggle Notebook

The main Kaggle notebook (`notebooks/kaggle_submission.ipynb`) provides a complete pipeline for the competition:

1. Data loading and validation
2. Model training and inference
3. Solution generation
4. Submission file creation
5. Results analysis and visualization

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with coverage
pytest --cov=src tests/
```

## Model Architecture Details

### Multi-Tensor System

The model operates on multiple tensor configurations simultaneously:
- Examples dimension: Individual ARC tasks
- Colors dimension: Color channels (0-9)
- Directions dimension: 8 directional channels
- Spatial dimensions: X and Y coordinates

### Training Process

1. **VAE Decoding**: Generate latent representations
2. **Multi-scale Processing**: Process at different tensor scales
3. **Directional Operations**: Apply cummax and shift operations
4. **Communication**: Share information across tensor dimensions
5. **Output Generation**: Predict color logits and spatial masks

### Loss Function

- **KL Divergence**: Regularizes latent representations
- **Reconstruction Error**: Cross-entropy loss for color prediction
- **Spatial Loss**: Mask prediction for output boundaries

## Performance

- **Memory Usage**: ~100-500MB per task (GPU)
- **Training Time**: ~3-5 minutes per task (2000 iterations)
- **Accuracy**: Competitive performance on ARC-AGI benchmark

## Acknowledgments

- ARC-AGI competition organizers
- PyTorch team for the deep learning framework
- Research community for inspiration and techniques

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{arc-agi-solver,
  title={Neural Network Approach to ARC-AGI Tasks},
  author={Your Name},
  year={2025},
  url={https://github.com/example/arc-agi-solver}
}
```

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch `(git checkout -b feature/your-feature)`.
3. Commit your changes `(git commit -m "Add your feature")`.
4. Push to the branch `(git push origin feature/your-feature)`.
5. Open a Pull Request.
6. Please include tests for new features and ensure existing tests pass. Follow PEP 8 style guidelines.

## License
This project is licensed under the MIT License.