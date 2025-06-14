
## Getting Started

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd arc-agi-solver
   make setup
   ```

2. **Verify Installation**
   ```bash
   make kaggle-test
   make test-unit
   ```

## Development Workflow

### 1. Code Changes
- Make changes to source code in `src/`
- Add tests in `tests/`
- Update documentation as needed

### 2. Testing
```bash
# Run specific test types
make test-unit           # Fast unit tests
make test-integration    # Integration tests
make test-slow          # Long-running tests
make test-coverage      # With coverage report
```

### 3. Code Quality
```bash
make format             # Auto-format code
make lint              # Check code quality
make check             # Full quality check
```

### 4. Experimentation
```bash
# Quick experiment (5 tasks, 2 minutes each)
make experiment

# Larger experiment (50 tasks)
make experiment-full

# Analyze results
make analyze
```

## Project Structure Details

### Source Code (`src/`)
- **arc_compressor.py**: Main VAE model implementation
- **multitensor_systems.py**: Multi-dimensional tensor handling
- **layers.py**: Custom neural network layers
- **initializers.py**: Weight initialization utilities
- **preprocessing.py**: Data preprocessing and task handling
- **solution_selection.py**: Solution tracking and logging
- **train.py**: Training loop implementation
- **visualization.py**: Plotting and visualization
- **solve_task.py**: Task solving orchestration

### Testing (`tests/`)
- **Unit tests**: Test individual components
- **Integration tests**: Test full pipeline
- **GPU tests**: Require CUDA-capable hardware
- **Slow tests**: Long-running tests

### Scripts (`scripts/`)
- **download_data.py**: Dataset management
- **run_training.py**: Local training experiments
- **generate_submission.py**: Kaggle submission generation
- **analyze_results.py**: Results analysis and visualization

## Model Architecture

### Multi-Tensor System
The core innovation is the multi-tensor system that handles different dimensional configurations:

```python
# Dimension meanings:
# [examples, colors, directions, x, y]
dims = [1, 1, 0, 1, 1]  # examples + colors + spatial
dims = [1, 0, 1, 1, 1]  # examples + directions + spatial
```

### Layer Sequence
1. **Decode Latents**: VAE decoder with KL loss
2. **Share Up**: Multi-tensor communication (bottom-up)
3. **Softmax**: Attention over dimension combinations
4. **Cummax/Shift**: Directional spatial operations
5. **Direction Share**: Communication between directions
6. **Nonlinear**: SiLU activation with residuals
7. **Share Down**: Multi-tensor communication (top-down)
8. **Normalize**: Layer normalization

### Training Process
- **Iterations**: 1500-2000 per task
- **Loss**: KL divergence + reconstruction error
- **Optimizer**: Adam with β=(0.5, 0.9)
- **Time**: ~3-5 minutes per task

## Debugging Tips

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Import errors**: Check Python path and dependencies
3. **Test failures**: Run with `-v` flag for details

### Debugging Commands
```bash
# Test single task with debug info
python scripts/run_training.py --max-tasks 1 --time-limit 60

# Check GPU memory
nvidia-smi

# Profile memory usage
python -m memory_profiler scripts/run_training.py
```

### Logging
- Model outputs saved to `outputs/`
- Plots saved to `outputs/plots/`
- Detailed logs in training scripts

## Performance Optimization

### Memory Management
- Use `torch.cuda.empty_cache()` between tasks
- Monitor peak memory with `torch.cuda.max_memory_allocated()`
- Consider gradient checkpointing for large models

### Speed Optimization
- Use multiple GPUs with `max_workers`
- Reduce iterations for faster experiments
- Profile with `torch.profiler`

## Contributing Guidelines

### Code Style
- Use Black for formatting: `make format`
- Follow PEP 8 conventions
- Add type hints where possible
- Document functions with docstrings

### Testing Requirements
- Add tests for new functionality
- Maintain >80% test coverage
- Include both unit and integration tests
- Test edge cases and error conditions

### Pull Request Process
1. Create feature branch
2. Make changes with tests
3. Run `make check`
4. Update documentation
5. Submit PR with description

## Kaggle Submission

### Preparation
1. **Test locally**: `make experiment`
2. **Verify data**: `make data-verify`
3. **Check imports**: `make kaggle-test`

### Submission Process
1. **Generate submission**: `make submission`
2. **Upload to Kaggle**: Copy notebook and data
3. **Run notebook**: Ensure clean execution
4. **Submit**: Use generated `submission.json`

### Optimization for Competition
- Reduce time limits for faster iteration
- Use fewer training iterations if needed
- Monitor memory usage carefully
- Include error handling for robustness

## Advanced Topics

### Custom Layers
Add new layers in `layers.py`:
```python
@multitensor_systems.multify
@add_residual
def my_custom_layer(dims, x):
    # Implementation here
    return processed_x
```

### Model Architecture Changes
Modify `ARCCompressor.__init__()`:
- Add new layer types
- Change channel dimensions
- Adjust layer sequence

### Multi-GPU Scaling
- Increase `max_workers` in solve functions
- Balance memory across GPUs
- Monitor GPU utilization