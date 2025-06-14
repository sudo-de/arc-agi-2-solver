import os
import json
import urllib.request
import zipfile
from pathlib import Path
import argparse

def download_file(url: str, filepath: str) -> None:
    """Download file from URL to filepath."""
    print(f"Downloading {url} to {filepath}")
    urllib.request.urlretrieve(url, filepath)
    print(f"Downloaded {filepath}")

def setup_data_directory(data_dir: str = "data") -> None:
    """Setup data directory structure."""
    data_path = Path(data_dir)
    
    # Create directories
    (data_path / "raw").mkdir(parents=True, exist_ok=True)
    (data_path / "processed").mkdir(parents=True, exist_ok=True)
    
    print(f"Created data directory structure at {data_path.absolute()}")

def download_arc_data(data_dir: str = "data") -> None:
    """Download ARC-AGI dataset files."""
    
    # Note: These URLs are placeholders - replace with actual ARC-AGI URLs
    base_url = "https://github.com/fchollet/ARC/raw/master/data"
    
    files_to_download = [
        "training_challenges.json",
        "training_solutions.json", 
        "evaluation_challenges.json",
        "evaluation_solutions.json",
        "test_challenges.json"
    ]
    
    data_path = Path(data_dir) / "raw"
    
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        filepath = data_path / f"arc-agi_{filename}"
        
        try:
            download_file(url, str(filepath))
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            # Create placeholder files for development
            create_placeholder_file(filepath, filename)

def create_placeholder_file(filepath: Path, filename: str) -> None:
    """Create placeholder data files for development."""
    print(f"Creating placeholder file: {filepath}")
    
    # Create minimal valid ARC task data
    if "challenges" in filename:
        placeholder_data = {
            "placeholder_task": {
                "train": [
                    {
                        "input": [[0, 1], [1, 0]],
                        "output": [[1, 0], [0, 1]]
                    }
                ],
                "test": [
                    {"input": [[0, 0], [1, 1]]}
                ]
            }
        }
    else:  # solutions file
        placeholder_data = {
            "placeholder_task": [[[1, 1], [0, 0]]]
        }
    
    with open(filepath, 'w') as f:
        json.dump(placeholder_data, f, indent=2)

def verify_data(data_dir: str = "data") -> bool:
    """Verify that all required data files exist and are valid."""
    data_path = Path(data_dir) / "raw"
    
    required_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_training_solutions.json",
        "arc-agi_evaluation_challenges.json", 
        "arc-agi_evaluation_solutions.json",
        "arc-agi_test_challenges.json"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = data_path / filename
        if not filepath.exists():
            print(f"Missing file: {filepath}")
            all_exist = False
        else:
            # Verify JSON is valid
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"✓ {filename}: {len(data)} tasks")
            except json.JSONDecodeError as e:
                print(f"✗ {filename}: Invalid JSON - {e}")
                all_exist = False
    
    return all_exist

def main():
    parser = argparse.ArgumentParser(description="Download ARC-AGI dataset")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    
    args = parser.parse_args()
    
    if args.verify_only:
        if verify_data(args.data_dir):
            print("All data files are present and valid")
        else:
            print("Some data files are missing or invalid")
        return
    
    print("Setting up ARC-AGI dataset...")
    setup_data_directory(args.data_dir)
    download_arc_data(args.data_dir)
    
    if verify_data(args.data_dir):
        print("Dataset setup completed successfully!")
    else:
        print("Dataset setup completed with some issues.")

if __name__ == "__main__":
    main()