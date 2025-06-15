import os
import json
import argparse
import sys
from pathlib import Path
from typing import Optional


def setup_data_directory(data_dir: str = "data") -> Path:
    """Create and setup data directory structure"""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    print(f"📁 Data directory: {data_path.absolute()}")
    return data_path


def organize_existing_files(source_dir: str, data_dir: Path) -> bool:
    """Organize existing ARC files from a source directory"""
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"✗ Source directory {source_dir} does not exist")
        return False
    # Expected filenames
    expected_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_training_solutions.json", 
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json",
        "arc-agi_test_challenges.json"
    ]
    found_files = 0
    for filename in expected_files:
        source_file = source_path / filename
        target_file = data_dir / filename
        if source_file.exists():
            try:
                import shutil
                shutil.copy2(source_file, target_file)
                print(f"✓ Copied {filename}")
                found_files += 1
            except Exception as e:
                print(f"✗ Failed to copy {filename}: {e}")
        else:
            print(f"⚠ File not found: {filename}")
    print(f"\n📊 Organized {found_files}/{len(expected_files)} files")
    return found_files > 0


def validate_data_files(data_dir: Path) -> bool:
    """Validate that downloaded data files are correct"""
    print("\n🔍 Validating data files...")
    required_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_training_solutions.json",
        "arc-agi_evaluation_challenges.json", 
        "arc-agi_evaluation_solutions.json",
        "arc-agi_test_challenges.json"
    ]
    validation_results = {}
    for filename in required_files:
        filepath = data_dir / filename
        validation_results[filename] = {
            'exists': filepath.exists(),
            'size': filepath.stat().st_size if filepath.exists() else 0,
            'valid_json': False,
            'task_count': 0
        }
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                validation_results[filename]['valid_json'] = True
                validation_results[filename]['task_count'] = len(data)
                print(f"✓ {filename}: {len(data)} tasks, {filepath.stat().st_size / 1024:.1f} KB")
            except json.JSONDecodeError:
                print(f"✗ {filename}: Invalid JSON format")
            except Exception as e:
                print(f"✗ {filename}: Error reading file - {e}")
        else:
            print(f"✗ {filename}: File not found")
    # Check if we have minimum required files
    essential_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_test_challenges.json"
    ]
    has_essential = all(
        validation_results[f]['exists'] and validation_results[f]['valid_json'] 
        for f in essential_files
    )
    if has_essential:
        print("✓ Essential data files validated successfully")
        print("\n📊 Dataset Summary:")
        for filename, results in validation_results.items():
            if results['valid_json']:
                split_name = filename.split('_')[1]  # training/evaluation/test
                file_type = 'solutions' if 'solutions' in filename else 'challenges'
                print(f"  {split_name.title()} {file_type}: {results['task_count']} tasks")
        return True
    else:
        print("✗ Essential data files missing or invalid")
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Organize and validate ARC-AGI data files")
    parser.add_argument('--data-dir', default='data', help='Data directory path (default: data)')
    parser.add_argument('--source-dir', default='data/raw', help='Source directory for organize method')
    args = parser.parse_args()
    print("🎯 ARC-AGI Data Organize/Validate Script")
    print("=" * 40)
    # Setup data directory
    data_dir = setup_data_directory(args.data_dir)
    # Organize files from source_dir
    organize_existing_files(args.source_dir, data_dir)
    # Validate files
    validation_success = validate_data_files(data_dir)
    if validation_success:
        print("\n🎉 Data organization and validation completed successfully!")
        print(f"Data location: {data_dir.absolute()}")
    else:
        print("\n✗ Data organization/validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()