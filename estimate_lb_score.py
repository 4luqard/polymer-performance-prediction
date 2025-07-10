#!/usr/bin/env python3
"""
Estimate LB score based on CV score and historical correlation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

def estimate_lb_from_cv(cv_score, history_file="cv_lb_tracking.csv"):
    """Estimate LB score from CV score"""
    try:
        history = pd.read_csv(history_file)
    except:
        print("No history file found. Using default ratio.")
        # Based on our one data point: CV=0.012, LB=0.158
        return cv_score * 13.17
    
    if len(history) < 2:
        # Use simple ratio
        ratio = history['lb_score'].iloc[0] / history['cv_score'].iloc[0]
        return cv_score * ratio
    
    # Use linear regression
    lr = LinearRegression()
    lr.fit(history[['cv_score']], history['lb_score'])
    
    lb_pred = lr.predict([[cv_score]])[0]
    
    print(f"Estimation based on {len(history)} historical submissions")
    print(f"Model: LB = {lr.coef_[0]:.2f} * CV + {lr.intercept_:.4f}")
    
    return lb_pred

def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_lb_score.py <cv_score>")
        print("   or: python estimate_lb_score.py auto  (runs CV first)")
        return
    
    if sys.argv[1] == "auto":
        # Run CV first
        import subprocess
        result = subprocess.run([sys.executable, "cross_validation.py"], 
                              capture_output=True, text=True)
        
        cv_score = None
        for line in result.stdout.split('\n'):
            if "Overall Competition Metric (wMAE):" in line:
                try:
                    cv_score = float(line.split(':')[1].split('(')[0].strip())
                    break
                except:
                    pass
        
        if cv_score is None:
            print("Could not extract CV score")
            return
    else:
        cv_score = float(sys.argv[1])
    
    print(f"\nCV Score: {cv_score:.4f}")
    lb_estimate = estimate_lb_from_cv(cv_score)
    print(f"Estimated LB Score: {lb_estimate:.4f}")
    
    # Compare with our baseline
    print(f"\nComparison with baseline:")
    print(f"  Baseline CV: 0.0120 → LB: 0.1580")
    print(f"  Your CV: {cv_score:.4f} → LB: ~{lb_estimate:.4f}")
    
    if cv_score < 0.0120:
        print("  ✓ Better than baseline!")
    else:
        print("  ✗ Worse than baseline")

if __name__ == "__main__":
    main()