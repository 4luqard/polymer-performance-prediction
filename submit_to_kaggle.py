#!/usr/bin/env python3
"""
Submit predictions to Kaggle NeurIPS Open Polymer Prediction 2025
This script handles both kernel creation and submission for code competitions

SECURITY NOTE:
- This script requires Kaggle API credentials at ~/.kaggle/kaggle.json
- NEVER commit kaggle.json to version control
- The .gitignore file is configured to exclude sensitive files
- All temporary submission files are cleaned up after use
"""

import os
import json
import subprocess
import sys
import time
import shutil
from datetime import datetime
import pandas as pd

# Check if kaggle API is available
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE_API = True
except ImportError:
    HAS_KAGGLE_API = False
    print("Warning: Kaggle API not installed. Will use CLI instead.")


def ensure_kaggle_auth():
    """Ensure Kaggle API credentials are available"""
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        print("\n" + "="*60)
        print("KAGGLE API CREDENTIALS NOT FOUND")
        print("="*60)
        print("\nTo use this script, you need to set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New Token'")
        print("3. This will download kaggle.json")
        print("4. Place the file at: ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nIMPORTANT: Never commit kaggle.json to version control!")
        print("="*60 + "\n")
        raise FileNotFoundError(
            "Kaggle credentials not found. Please set up ~/.kaggle/kaggle.json"
        )
    return True


def create_kernel_metadata(kernel_name="neurips-polymer-submission"):
    """Create kernel metadata for submission"""
    metadata = {
        "id": f"{{username}}/{kernel_name}",  # Will be filled by Kaggle
        "title": f"NeurIPS Polymer Prediction - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "code_file": "model.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["neurips-open-polymer-prediction-2025"],
        "kernel_sources": [],
        "model_sources": []
    }
    
    # Save metadata
    kernel_dir = "/workspace/kaggle/neurips-open-polymer-prediction-2025/submission_kernel"
    os.makedirs(kernel_dir, exist_ok=True)
    
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Copy model.py to kernel directory
    subprocess.run([
        "cp", 
        "/workspace/kaggle/neurips-open-polymer-prediction-2025/model.py",
        os.path.join(kernel_dir, "model.py")
    ], check=True)
    
    # Copy necessary utility files
    utils_dir = os.path.join(kernel_dir, "src")
    os.makedirs(utils_dir, exist_ok=True)
    subprocess.run([
        "cp",
        "/workspace/kaggle/neurips-open-polymer-prediction-2025/src/competition_metric.py",
        os.path.join(utils_dir, "competition_metric.py")
    ], check=True)
    
    return kernel_dir


def submit_via_cli(kernel_dir, message="Automated submission for CV/LB sync"):
    """Submit using Kaggle CLI"""
    try:
        # Find kaggle executable
        kaggle_cmd = "kaggle"
        if not subprocess.run(["which", kaggle_cmd], capture_output=True).returncode == 0:
            # Try local installation
            local_kaggle = os.path.expanduser("~/.local/bin/kaggle")
            if os.path.exists(local_kaggle):
                kaggle_cmd = local_kaggle
            else:
                print("Kaggle CLI not found. Please install: pip install kaggle")
                return False
        
        # First, push the kernel
        print("Pushing kernel to Kaggle...")
        push_result = subprocess.run(
            [kaggle_cmd, "kernels", "push", "-p", kernel_dir],
            capture_output=True,
            text=True
        )
        
        if push_result.returncode != 0:
            print(f"Error pushing kernel: {push_result.stderr}")
            return False
        
        print("Kernel pushed successfully!")
        print(push_result.stdout)
        
        # Extract kernel slug from output if available
        # The output typically contains the kernel URL
        
        # For code competitions, the kernel needs to run first
        # Then we can submit the output
        print("\nNote: For code competitions, the kernel must complete running before submission.")
        print("Please check the Kaggle website for kernel status and submit manually if needed.")
        
        return True
        
    except Exception as e:
        print(f"Error during submission: {e}")
        return False


def submit_via_api(message="Automated submission for CV/LB sync"):
    """Submit using Kaggle Python API"""
    if not HAS_KAGGLE_API:
        print("Kaggle API not available, falling back to CLI method")
        return False
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # For code competitions, we need to create and push a kernel
        print("Creating submission kernel...")
        kernel_dir = create_kernel_metadata()
        
        # Push kernel using API (if available in newer versions)
        # Note: This might require manual submission through website
        print("Note: Code competition submission via API may require manual steps.")
        print("Please check https://www.kaggle.com/code for your pushed kernel.")
        
        return submit_via_cli(kernel_dir, message)
        
    except Exception as e:
        print(f"API submission failed: {e}")
        return False


def generate_submission():
    """Run the model to generate a fresh submission file"""
    print("Generating fresh submission...")
    try:
        result = subprocess.run(
            [sys.executable, "/workspace/kaggle/neurips-open-polymer-prediction-2025/model.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error running model: {result.stderr}")
            return False
            
        print("Submission generated successfully!")
        return True
        
    except Exception as e:
        print(f"Error generating submission: {e}")
        return False


def get_latest_cv_score():
    """Extract the latest CV score from diagnostic files"""
    diagnostics_dir = "/workspace/kaggle/neurips-open-polymer-prediction-2025/output/cv_diagnostics"
    
    if not os.path.exists(diagnostics_dir):
        return None
    
    # Get the latest diagnostic file
    json_files = [f for f in os.listdir(diagnostics_dir) if f.endswith('.json')]
    if not json_files:
        return None
    
    latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(diagnostics_dir, f)))
    
    with open(os.path.join(diagnostics_dir, latest_file), 'r') as f:
        data = json.load(f)
        return data.get('overall_score')


def cleanup_submission_files():
    """Clean up temporary submission files"""
    temp_dirs = ["submission_kernel"]
    temp_files = ["kernel-metadata.json"]
    
    for dir_path in temp_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"✓ Cleaned up {dir_path}")
    
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Cleaned up {file_path}")


def main():
    """Main submission workflow"""
    print("=== Kaggle Competition Submission Tool ===")
    print("Competition: NeurIPS Open Polymer Prediction 2025")
    print()
    
    try:
        # Ensure authentication
        try:
            ensure_kaggle_auth()
            print("✓ Kaggle authentication configured")
        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            return 1
        
        # Get latest CV score
        cv_score = get_latest_cv_score()
        if cv_score:
            print(f"✓ Latest CV score: {cv_score:.4f}")
        else:
            print("! No CV score found")
        
        # Generate submission
        if not generate_submission():
            print("✗ Failed to generate submission")
            return 1
        
        # Create kernel directory
        kernel_dir = create_kernel_metadata()
        print(f"✓ Kernel directory created: {kernel_dir}")
        
        # Submit
        submission_message = f"CV Score: {cv_score:.4f} - Sync test" if cv_score else "Sync test submission"
        
        if submit_via_cli(kernel_dir, submission_message):
            print("\n✓ Submission process completed!")
            print("\nNext steps:")
            print("1. Check https://www.kaggle.com/code to verify kernel upload")
            print("2. Wait for kernel to complete running")
            print("3. Submit the kernel output to the competition")
            print("4. Compare public LB score with CV score to verify sync")
        else:
            print("\n✗ Submission failed")
            return 1
        
        return 0
        
    finally:
        # Always clean up temporary files
        print("\nCleaning up temporary files...")
        cleanup_submission_files()


if __name__ == "__main__":
    sys.exit(main())