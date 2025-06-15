import pytest
from pathlib import Path
import json
import logging
from scripts.generate_submission import (
    setup_logging,
    load_test_challenges,
    load_existing_submission,
    validate_submission_format,
    generate_default_submission,
    calculate_submission_statistics,
    save_submission,
    generate_model_predictions,
    enhance_submission_with_heuristics
)

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing"""
    return tmp_path

@pytest.fixture
def sample_test_challenges():
    """Create sample test challenges"""
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
def sample_submission():
    """Create a sample submission"""
    return {
        "task1": [
            {"attempt_1": [[1, 0], [0, 1]], "attempt_2": [[0, 1], [1, 0]]}
        ]
    }

def test_setup_logging(temp_dir):
    """Test logging setup"""
    logger = setup_logging(temp_dir)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "submission_generator"
    assert logger.level == logging.INFO

def test_load_test_challenges(temp_dir, sample_test_challenges):
    """Test loading test challenges"""
    # Create test challenges file
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    with open(data_dir / "arc-agi_test_challenges.json", 'w') as f:
        json.dump(sample_test_challenges, f)
    
    challenges = load_test_challenges(data_dir)
    assert isinstance(challenges, dict)
    assert "task1" in challenges

def test_load_existing_submission(temp_dir, sample_submission):
    """Test loading existing submission"""
    # Create submission file
    submission_file = temp_dir / "submission.json"
    with open(submission_file, 'w') as f:
        json.dump(sample_submission, f)
    
    submission = load_existing_submission(submission_file)
    assert isinstance(submission, dict)
    assert "task1" in submission

def test_validate_submission_format(temp_dir, sample_test_challenges, sample_submission):
    """Test submission format validation"""
    logger = setup_logging(temp_dir)
    is_valid, errors = validate_submission_format(sample_submission, sample_test_challenges, logger)
    assert is_valid
    assert len(errors) == 0

def test_generate_default_submission(sample_test_challenges):
    """Test default submission generation"""
    submission = generate_default_submission(sample_test_challenges)
    assert isinstance(submission, dict)
    assert "task1" in submission
    assert len(submission["task1"]) == 1
    assert "attempt_1" in submission["task1"][0]
    assert "attempt_2" in submission["task1"][0]

def test_calculate_submission_statistics(sample_test_challenges, sample_submission):
    """Test submission statistics calculation"""
    stats = calculate_submission_statistics(sample_submission, sample_test_challenges)
    assert isinstance(stats, dict)
    assert "total_tasks" in stats
    assert "total_attempts" in stats
    assert "average_attempts_per_task" in stats

def test_save_submission(temp_dir, sample_submission):
    """Test saving submission"""
    output_file = temp_dir / "submission.json"
    logger = setup_logging(temp_dir)
    save_submission(sample_submission, output_file, logger)
    assert output_file.exists()
    with open(output_file, 'r') as f:
        saved_data = json.load(f)
        assert saved_data == sample_submission

def test_generate_model_predictions(temp_dir, sample_test_challenges):
    """Test model predictions generation"""
    logger = setup_logging(temp_dir)
    config = {
        "model_path": "models/best_model.pth",
        "device": "cpu",
        "batch_size": 1
    }
    predictions = generate_model_predictions(sample_test_challenges, config, logger)
    assert isinstance(predictions, dict)
    assert "task1" in predictions

def test_enhance_submission_with_heuristics(temp_dir, sample_test_challenges, sample_submission):
    """Test submission enhancement with heuristics"""
    logger = setup_logging(temp_dir)
    enhanced = enhance_submission_with_heuristics(sample_submission, sample_test_challenges, logger)
    assert isinstance(enhanced, dict)
    assert "task1" in enhanced
    assert len(enhanced["task1"]) == 2

def test_load_test_challenges_missing(temp_dir):
    """Test loading missing test challenges"""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_test_challenges(data_dir)

def test_validate_submission_format_invalid(temp_dir, sample_test_challenges):
    """Test validation of invalid submission format"""
    logger = setup_logging(temp_dir)
    invalid_submission = {
        "task1": [{"invalid_key": [[1, 0], [0, 1]]}]
    }
    is_valid, errors = validate_submission_format(invalid_submission, sample_test_challenges, logger)
    assert not is_valid
    assert len(errors) > 0

def test_save_submission_error(temp_dir, sample_submission):
    """Test saving submission with error"""
    logger = setup_logging(temp_dir)
    output_file = temp_dir / "invalid" / "submission.json"
    save_submission(sample_submission, output_file, logger)
    assert output_file.exists() 