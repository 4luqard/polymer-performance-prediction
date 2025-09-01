#!/usr/bin/env python3
import subprocess
import time
import re

def run_cv_with_deepchem():
    """Run CV with DeepChem tokenizer"""
    print("Running CV with DeepChem tokenizer...")
    start_time = time.time()
    
    # Run CV with timeout for fast test
    result = subprocess.run(
        ["python3", "model.py", "--cv-only"],
        capture_output=True,
        text=True,
        timeout=180  # 3 minutes timeout
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract CV score from output
    cv_score = None
    for line in result.stdout.split('\n'):
        if 'Overall:' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                cv_score = float(match.group(1))
                break
    
    print(f"CV with DeepChem completed in {elapsed_time:.1f} seconds")
    if cv_score:
        print(f"DeepChem CV Score: {cv_score:.4f}")
    else:
        print("Could not extract CV score from output")
        print("Output tail:", result.stdout[-500:])
    
    return cv_score

if __name__ == "__main__":
    baseline_score = 0.0556  # From baseline test
    deepchem_score = run_cv_with_deepchem()
    
    if deepchem_score:
        improvement = baseline_score - deepchem_score
        pct_change = (improvement / baseline_score) * 100
        print(f"\n=== Comparison ===")
        print(f"Baseline CV Score: {baseline_score:.4f}")
        print(f"DeepChem CV Score: {deepchem_score:.4f}")
        print(f"Improvement: {improvement:.4f} ({pct_change:.1f}%)")