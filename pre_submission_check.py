#!/usr/bin/env python3
"""
Pre-submission checker to estimate LB score from CV
"""

import sys
import subprocess
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def run_cv():
    """Run cross-validation and get score"""
    print("Running cross-validation...")
    result = subprocess.run([sys.executable, "cross_validation.py"], 
                          capture_output=True, text=True)
    
    # Parse CV score
    for line in result.stdout.split('\n'):
        if "Overall Competition Metric (wMAE):" in line:
            try:
                cv_score = float(line.split(':')[1].split('(')[0].strip())
                return cv_score
            except:
                pass
    return None

def load_history():
    """Load CV/LB tracking history"""
    try:
        return pd.read_csv("cv_lb_tracking.csv")
    except:
        return None

def predict_lb(cv_score, history):
    """Predict LB score from CV score using historical data"""
    if len(history) < 3:
        # Simple ratio-based prediction
        avg_ratio = history['lb_score'].mean() / history['cv_score'].mean()
        return cv_score * avg_ratio
    else:
        # Linear regression
        lr = LinearRegression()
        lr.fit(history[['cv_score']], history['lb_score'])
        return lr.predict([[cv_score]])[0]

def main():
    """Main pre-submission check"""
    print("=== Pre-Submission Check ===\n")
    
    # Run CV
    cv_score = run_cv()
    if cv_score is None:
        print("Failed to get CV score")
        return
    
    print(f"\nCV Score: {cv_score:.4f}")
    
    # Load history
    history = load_history()
    if history is None or len(history) == 0:
        print("\nNo historical data available for LB prediction")
        print("Submit this model and track the result with:")
        print(f"  python3 track_cv_lb.py add {cv_score:.4f} <lb_score> 'description'")
        return
    
    # Predict LB
    predicted_lb = predict_lb(cv_score, history)
    print(f"\nPredicted LB Score: {predicted_lb:.4f}")
    
    # Show confidence based on historical data
    if len(history) >= 5:
        # Calculate prediction interval
        cv_values = history['cv_score'].values
        lb_values = history['lb_score'].values
        
        lr = LinearRegression()
        lr.fit(cv_values.reshape(-1, 1), lb_values)
        
        residuals = lb_values - lr.predict(cv_values.reshape(-1, 1))
        std_residual = np.std(residuals)
        
        print(f"95% Confidence Interval: [{predicted_lb - 2*std_residual:.4f}, {predicted_lb + 2*std_residual:.4f}]")
    
    # Recommendations
    print("\n=== Recommendations ===")
    
    # Check against best score
    best_lb = history['lb_score'].min()
    if predicted_lb < best_lb:
        print(f"✓ Predicted score ({predicted_lb:.4f}) is better than current best ({best_lb:.4f})")
        print("  → Worth submitting!")
    else:
        print(f"✗ Predicted score ({predicted_lb:.4f}) is worse than current best ({best_lb:.4f})")
        print("  → Consider improving the model first")
    
    # Show recent history
    print("\n=== Recent Submissions ===")
    print(history[['cv_score', 'lb_score', 'notes']].tail(5).to_string(index=False))

if __name__ == "__main__":
    main()