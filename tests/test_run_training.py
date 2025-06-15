import pytest
from pathlib import Path
import json
import logging
import torch
import os
from scripts.run_training import (
    TrainingConfig,
    setup_logging,
    setup_device,
    load_tasks,
    enhanced_solve_task,
    run_training,
    save_results,
    generate_summary_stats
)

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing"""
    return tmp_path

@pytest.fixture
def sample_training_data():
    """Create sample training data"""
    return {
        "task1": {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
            ],
            "test": [
                {"input": [[0, 1], [1, 0]]}
            ]
        }
    }

@pytest.fixture
def config():
    """Create a test configuration"""
    return TrainingConfig(
        data_dir="data",
        split="training",
        task_limit=1,
        max_iterations=10,
        time_limit_per_task=5,
        learning_rate=0.01,
        adam_betas=(0.5, 0.9),
        early_stopping_threshold=5,
        convergence_check_interval=2,
        memory_cleanup_interval=2,
        progress_report_interval=1,
        results_dir="results",
        save_plots=False,
        save_metrics=True,
        save_solutions=True,
        device="cpu",
        random_seed=42
    )

def test_setup_logging(temp_dir):
    """Test logging setup"""
    logger = setup_logging(temp_dir)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "arc_training"
    assert logger.level == logging.INFO

def test_setup_device(config):
    """Test device setup"""
    device = setup_device(config)
    assert device in ["cpu", "cuda"]
    if torch.cuda.is_available():
        assert device == "cuda"
    else:
        assert device == "cpu"

def test_load_tasks(temp_dir, sample_training_data, config):
    """Test task loading"""
    # Create training data file
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    with open(data_dir / "arc-agi_training_challenges.json", 'w') as f:
        json.dump(sample_training_data, f)
    
    # Update config to use the temporary data directory
    config.data_dir = str(data_dir)
    
    logger = setup_logging(temp_dir)
    tasks = load_tasks(config, logger)
    assert len(tasks) > 0

def test_enhanced_solve_task(temp_dir, sample_training_data, config):
    """Test enhanced task solving"""
    logger = setup_logging(temp_dir)
    task_name = "task1"
    problem_data = sample_training_data[task_name]
    
    solutions, metrics = enhanced_solve_task(task_name, problem_data, config, logger)
    assert isinstance(solutions, list)
    assert isinstance(metrics, dict)
    assert len(solutions) > 0
    assert "accuracy" in metrics

def test_run_training(temp_dir, sample_training_data, config):
    """Test training run"""
    # Create training data file
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    with open(data_dir / "arc-agi_training_challenges.json", 'w') as f:
        json.dump(sample_training_data, f)
    
    # Update config to use the temporary data directory
    config.data_dir = str(data_dir)
    config.results_dir = str(temp_dir / "results")
    
    logger = setup_logging(temp_dir)
    solutions, metrics = run_training(config, logger)
    assert isinstance(solutions, dict)
    assert isinstance(metrics, dict)
    assert "task1" in solutions
    assert "accuracy" in metrics

def test_save_results(temp_dir, sample_training_data, config):
    """Test saving results"""
    # Create sample solutions and metrics
    solutions = {
        "task1": [[[1, 0], [0, 1]]]
    }
    metrics = {
        "task1": {
            "accuracy": 0.8,
            "time_taken": 10.5
        }
    }
    # Update config
    config.results_dir = str(temp_dir / "results")
    logger = setup_logging(temp_dir)
    save_results(solutions, metrics, config, logger)
    # Check if files were created
    assert (temp_dir / "results" / "submission.json").exists()
    assert (temp_dir / "results" / "detailed_metrics.json").exists()

def test_generate_summary_stats():
    """Test summary statistics generation"""
    metrics = {
        "task1": {
            "accuracy": 0.8,
            "time_taken": 10.5,
            "iterations": 100
        }
    }
    stats = generate_summary_stats(metrics)
    assert isinstance(stats, dict)
    assert "total_tasks" in stats
    assert "successful_tasks" in stats
    assert "failed_tasks" in stats
    assert "convergence_rate" in stats
    assert "average_steps" in stats
    assert "average_time" in stats
    assert "average_memory" in stats
    assert "total_time" in stats

def test_setup_device_auto(config):
    """Test automatic device selection"""
    config.device = "auto"
    device = setup_device(config)
    assert device in ["cpu", "cuda"]

def test_load_tasks_empty(temp_dir, config):
    """Test loading tasks from empty directory"""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    config.data_dir = str(data_dir)
    
    logger = setup_logging(temp_dir)
    with pytest.raises(FileNotFoundError):
        load_tasks(config, logger)

def test_save_results_error(temp_dir, config):
    """Test saving results with error"""
    solutions = {"task1": [[[1, 0], [0, 1]]]}
    metrics = {"accuracy": 0.8}
    config.results_dir = str(temp_dir / "invalid" / "results")
    
    logger = setup_logging(temp_dir)
    with pytest.raises(FileNotFoundError):
        save_results(solutions, metrics, config, logger)

class TestRunTraining:
    def setup_method(self):
        """Setup test environment before each test"""
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        self.results_dir = self.test_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create sample training data
        self.sample_data = {
            "arc-agi_training_challenges.json": {
                "train": [
                    {
                        "input": [[0, 1], [1, 0]],
                        "output": [[1, 0], [0, 1]]
                    }
                ]
            },
            "arc-agi_training_solutions.json": {
                "train": [
                    {
                        "input": [[0, 1], [1, 0]],
                        "output": [[1, 0], [0, 1]]
                    }
                ]
            }
        }
        
        for filename, data in self.sample_data.items():
            with open(self.test_dir / filename, 'w') as f:
                json.dump(data, f)

    def teardown_method(self):
        """Cleanup after each test"""
        if self.test_dir.exists():
            for file in self.test_dir.glob("**/*"):
                if file.is_file():
                    file.unlink()
            for dir in reversed(list(self.test_dir.glob("**/*"))):
                if dir.is_dir():
                    dir.rmdir()
            self.test_dir.rmdir()

    def test_training_config(self):
        """Test TrainingConfig initialization"""
        config = TrainingConfig(
            data_dir=str(self.test_dir),
            split="training",
            max_iterations=100,
            time_limit_per_task=30
        )
        assert config.data_dir == str(self.test_dir)
        assert config.split == "training"
        assert config.max_iterations == 100
        assert config.time_limit_per_task == 30

    def test_setup_logging(self):
        """Test logging setup"""
        logger = setup_logging(self.results_dir)
        assert logger is not None
        assert logger.name == "arc_training"

    def test_setup_device(self):
        """Test device setup"""
        config = TrainingConfig(device="auto")
        device = setup_device(config)
        assert device in ["cuda", "cpu"]

    def test_load_tasks(self):
        """Test task loading"""
        config = TrainingConfig(
            data_dir=str(self.test_dir),
            split="training"
        )
        logger = setup_logging(self.results_dir)
        tasks = load_tasks(config, logger)
        assert len(tasks) > 0
        assert all(isinstance(task, tuple) for task in tasks)
        assert all(len(task) == 2 for task in tasks)

    def test_enhanced_solve_task(self):
        """Test enhanced task solving"""
        config = TrainingConfig(
            data_dir=str(self.test_dir),
            split="training",
            max_iterations=10
        )
        logger = setup_logging(self.results_dir)
        
        # Create a sample task
        task_name = "test_task"
        problem_data = {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]]
                }
            ]
        }
        
        solutions, metrics = enhanced_solve_task(task_name, problem_data, config, logger)
        assert isinstance(solutions, list)
        assert isinstance(metrics, dict)

    def test_run_training(self):
        """Test full training run"""
        config = TrainingConfig(
            data_dir=str(self.test_dir),
            split="training",
            max_iterations=10,
            time_limit_per_task=30,
            results_dir=str(self.results_dir)
        )
        logger = setup_logging(self.results_dir)
        
        solutions, metrics = run_training(config, logger)
        assert isinstance(solutions, dict)
        assert isinstance(metrics, dict)

    def test_save_results(self):
        """Test saving training results"""
        config = TrainingConfig(
            data_dir=str(self.test_dir),
            results_dir=str(self.results_dir)
        )
        logger = setup_logging(self.results_dir)
        
        solutions = {"task1": [[[1, 0], [0, 1]]]}
        metrics = {"task1": {"accuracy": 0.8}}
        
        save_results(solutions, metrics, config, logger)
        
        # Verify files were created
        assert (self.results_dir / "submission.json").exists()
        assert (self.results_dir / "detailed_metrics.json").exists() 