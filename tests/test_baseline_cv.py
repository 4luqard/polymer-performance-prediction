#!/usr/bin/env python3
import subprocess
import time
import re

def run_baseline_cv():
    """Run baseline CV with current tokenizer"""
    print("Running baseline CV with current tokenizer...")
    start_time = time.time()
    
    # Run CV with timeout
    result = subprocess.run(
        ["python3", "model.py", "--cv-only"],
        capture_output=True,
        text=True,
        timeout=180  # 3 minutes timeout for fast test
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract CV score from output
    cv_score = None
    for line in result.stdout.split('\n'):
        if 'Final CV Score' in line or 'CV Score' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                cv_score = float(match.group(1))
                break
    
    print(f"Baseline CV completed in {elapsed_time:.1f} seconds")
    if cv_score:
        print(f"Baseline CV Score: {cv_score:.4f}")
    else:
        print("Could not extract CV score from output")
        print("Output tail:", result.stdout[-500:])
    
    return cv_score

if __name__ == "__main__":
    run_baseline_cv()