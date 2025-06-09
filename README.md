# ARC-AGI-2 Solver

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![pytest](https://github.com/sudo-de/arc-agi-2-solver/actions/workflows/pytest.yml/badge.svg)](https://github.com/sudo-de/arc-agi-2-solver/actions)

A Python-based solver for the **[Abstraction and Reasoning Challenge (ARC-AGI-2)](https://www.kaggle.com/competitions/arc-prize-2025)**, designed to tackle grid-based pattern recognition tasks using a variational autoencoder (VAE) approach. This project implements a modular, robust, and extensible framework for processing, training, and visualizing ARC tasks, with a focus on achieving high performance in the Kaggle competition.

## Features

- **Modular Architecture**: Organized into distinct modules for task processing (`arc_task_processor`), multi-tensor operations (`arc_multitensor`), model training (`arc_trainer`), and visualization (`arc_visualizer`).
- **VAE-Based Model**: Implements `ARCVAE` for encoding and decoding ARC grids, with support for multi-dimensional tensor systems.
- **Robust Testing**: Comprehensive pytest suite covering task processing, model training, and solution selection, ensuring reliability.
- **Visualization Tools**: Generates high-quality plots for task inputs, outputs, and predicted solutions using Matplotlib.
- **Submission-Ready**: Produces JSON submissions compatible with ARC-AGI-2 evaluation requirements.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/arc-agi-2-solver.git
   cd arc-agi-2-solver
   ```

2. **Set Up a Virtual Environment:**
   ```bash
    python -m venv venv
    source venv/bin/activate  # On MacOS
    venv\Scripts\activate # On Windows
    ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare Data:**
   - Place ARC-AGI-2 task files (e.g., arc-agi_test_challenges.json) in the data/ directory.
   - Update src/arc_task_processor.py to point to your data path if necessary.
5. **Running Tests**
   ```bash
   pytest tests/ -v --cov=src
   ```
## Project Structure
    arc-agi-2-solver/
    ├── src/
    │   ├── arc_task_processor.py  # Task loading and preprocessing
    │   ├── arc_multitensor.py     # Multi-dimensional tensor operations
    │   ├── arc_weight_initializer.py  # Weight initialization for VAE
    │   ├── arc_network_layers.py  # Neural network layers
    │   ├── arc_vae_decoder.py     # VAE model implementation
    │   ├── arc_solution_selector.py  # Solution tracking and selection
    │   ├── arc_visualizer.py      # Visualization of tasks and solutions
    │   ├── arc_trainer.py         # Training logic
    ├── tests/
    │   ├── conftest.py            # Pytest fixtures
    │   ├── test_*.py             # Test modules
        │   ├── data/
        │   │   └── sample_task.json  # Sample ARC task
    ├── data/                     # Place ARC task files here
    ├── plots/                    # Output directory for visualizations
    ├── requirements.txt          # Python dependencies
    ├── pytest.ini               # Pytest configuration
    ├── LICENSE                  # License file
    ├── README.md                # This file

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
