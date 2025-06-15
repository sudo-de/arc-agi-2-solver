import os
import json
import pytest
from pathlib import Path
from scripts.download_data import (
    setup_data_directory,
    organize_existing_files,
    validate_data_files
)

class TestDownloadData:
    def setup_method(self):
        """Setup test environment before each test"""
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        self.raw_dir = self.test_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

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

    def test_setup_data_directory(self):
        """Test data directory setup"""
        data_dir = setup_data_directory(str(self.test_dir))
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_organize_existing_files(self):
        """Test organizing existing files"""
        # Create sample files in raw directory
        expected_files = [
            "arc-agi_training_challenges.json",
            "arc-agi_training_solutions.json",
            "arc-agi_evaluation_challenges.json",
            "arc-agi_evaluation_solutions.json",
            "arc-agi_test_challenges.json"
        ]
        
        for filename in expected_files:
            with open(self.raw_dir / filename, 'w') as f:
                json.dump({"test": "data"}, f)

        # Test organizing files
        result = organize_existing_files(str(self.raw_dir), self.test_dir)
        assert result is True

        # Verify files were copied
        for filename in expected_files:
            assert (self.test_dir / filename).exists()

    def test_validate_data_files(self):
        """Test data file validation"""
        # Create valid JSON files
        expected_files = [
            "arc-agi_training_challenges.json",
            "arc-agi_training_solutions.json",
            "arc-agi_evaluation_challenges.json",
            "arc-agi_evaluation_solutions.json",
            "arc-agi_test_challenges.json"
        ]
        
        for filename in expected_files:
            with open(self.test_dir / filename, 'w') as f:
                json.dump({"test": "data"}, f)

        # Test validation
        result = validate_data_files(self.test_dir)
        assert result is True

    def test_create_sample_data(self):
        """Test creating sample data files"""
        # Create sample data files
        sample_data = {
            "arc-agi_training_challenges.json": {"train": [{"input": [[0]], "output": [[1]]}]},
            "arc-agi_training_solutions.json": {"train": [{"input": [[0]], "output": [[1]]}]},
            "arc-agi_evaluation_challenges.json": {"test": [{"input": [[0]]}]},
            "arc-agi_evaluation_solutions.json": {"test": [{"input": [[0]], "output": [[1]]}]},
            "arc-agi_test_challenges.json": {"test": [{"input": [[0]]}]}
        }

        for filename, data in sample_data.items():
            with open(self.test_dir / filename, 'w') as f:
                json.dump(data, f)

        # Validate the created files
        result = validate_data_files(self.test_dir)
        assert result is True 