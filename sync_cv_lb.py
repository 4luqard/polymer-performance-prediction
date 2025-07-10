#!/usr/bin/env python3
"""
Automated CV/LB synchronization script
Just input the LB score after submission and it handles everything
"""

import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import os

def run_cv():
    """Run cross-validation and extract score"""
    print("Running cross-validation...")
    result = subprocess.run([sys.executable, "cross_validation.py"], 
                          capture_output=True, text=True)
    
    # Parse CV score and individual scores
    cv_score = None
    individual_scores = {}
    
    for line in result.stdout.split('\n'):
        if "Overall Competition Metric (wMAE):" in line:
            try:
                cv_score = float(line.split(':')[1].split('(')[0].strip())
            except:
                pass
        # Parse individual scores
        elif "Tg:" in line and "±" in line:
            individual_scores['Tg'] = float(line.split(':')[1].split('(')[0].strip())
        elif "FFV:" in line and "±" in line:
            individual_scores['FFV'] = float(line.split(':')[1].split('(')[0].strip())
        elif "Tc:" in line and "±" in line:
            individual_scores['Tc'] = float(line.split(':')[1].split('(')[0].strip())
        elif "Density:" in line and "±" in line:
            individual_scores['Density'] = float(line.split(':')[1].split('(')[0].strip())
        elif "Rg:" in line and "±" in line:
            individual_scores['Rg'] = float(line.split(':')[1].split('(')[0].strip())
    
    return cv_score, individual_scores

def update_tracking(cv_score, lb_score, notes=""):
    """Update CV/LB tracking file"""
    history_file = "cv_lb_tracking.csv"
    
    # Load existing history
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
    else:
        history = pd.DataFrame(columns=['timestamp', 'cv_score', 'lb_score', 'difference', 'ratio', 'notes'])
    
    # Add new entry
    new_entry = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cv_score': cv_score,
        'lb_score': lb_score,
        'difference': cv_score - lb_score,
        'ratio': cv_score / lb_score if lb_score > 0 else np.nan,
        'notes': notes
    }])
    
    history = pd.concat([history, new_entry], ignore_index=True)
    history.to_csv(history_file, index=False)
    
    return history

def analyze_correlation(history):
    """Analyze CV/LB correlation and provide insights"""
    print("\n=== CV/LB Correlation Analysis ===")
    print(f"Total submissions tracked: {len(history)}")
    
    if len(history) < 2:
        print("Need at least 2 submissions for correlation analysis")
        return None
    
    # Basic statistics
    correlation = history['cv_score'].corr(history['lb_score'])
    print(f"\nCorrelation coefficient: {correlation:.4f}")
    
    avg_ratio = history['ratio'].mean()
    std_ratio = history['ratio'].std()
    print(f"Average CV/LB ratio: {avg_ratio:.4f} (±{std_ratio:.4f})")
    
    # Fit linear model
    if len(history) >= 3:
        lr = LinearRegression()
        lr.fit(history[['cv_score']], history['lb_score'])
        
        print(f"\nLinear model: LB = {lr.coef_[0]:.2f} * CV + {lr.intercept_:.4f}")
        
        # Calculate R²
        r2 = lr.score(history[['cv_score']], history['lb_score'])
        print(f"R² score: {r2:.4f}")
        
        return lr
    else:
        return None

def suggest_cv_adjustments(cv_score, lb_score, individual_cv_scores):
    """Suggest how to adjust CV to better match LB"""
    print("\n=== CV Adjustment Suggestions ===")
    
    # Calculate the gap
    gap_ratio = lb_score / cv_score
    print(f"\nCurrent gap: CV={cv_score:.4f} vs LB={lb_score:.4f} (ratio={gap_ratio:.2f})")
    
    if gap_ratio > 10:
        print("\n⚠️  HUGE GAP DETECTED - Possible issues:")
        print("1. CV metric calculation might be different from competition")
        print("2. Test set has very different distribution than train")
        print("3. Model is overfitting to training data patterns")
        
        print("\nRecommendations:")
        print("1. Double-check metric implementation against competition code")
        print("2. Use more conservative validation (e.g., time-based splits)")
        print("3. Add regularization or use simpler models")
        
        # Check which targets contribute most to the gap
        if individual_cv_scores:
            print("\nPer-target CV scores (may help identify problematic targets):")
            for target, score in individual_cv_scores.items():
                expected_lb_contribution = score * gap_ratio
                print(f"  {target}: CV={score:.4f} → Expected LB contribution ~{expected_lb_contribution:.4f}")

def predict_next_lb(cv_score, model):
    """Predict LB score for a given CV score"""
    if model is None:
        return None
    
    predicted_lb = model.predict([[cv_score]])[0]
    return predicted_lb

def main():
    if len(sys.argv) < 2:
        print("Usage: python sync_cv_lb.py <lb_score> [notes]")
        print("\nThis script will:")
        print("1. Run cross-validation automatically")
        print("2. Track the CV/LB pair")
        print("3. Analyze correlation")
        print("4. Suggest adjustments")
        return
    
    lb_score = float(sys.argv[1])
    notes = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Auto-tracked"
    
    print("=== CV/LB Synchronization ===\n")
    
    # Step 1: Run CV
    cv_score, individual_scores = run_cv()
    if cv_score is None:
        print("Failed to get CV score")
        return
    
    print(f"\nCV Score: {cv_score:.4f}")
    print(f"LB Score: {lb_score:.4f}")
    print(f"Gap Ratio: {lb_score/cv_score:.2f}x")
    
    # Step 2: Update tracking
    history = update_tracking(cv_score, lb_score, notes)
    print(f"\n✓ Tracked submission #{len(history)}")
    
    # Step 3: Analyze correlation
    model = analyze_correlation(history)
    
    # Step 4: Show recent history
    print("\n=== Recent Submissions ===")
    display_cols = ['timestamp', 'cv_score', 'lb_score', 'ratio', 'notes']
    print(history[display_cols].tail(5).to_string(index=False))
    
    # Step 5: Suggest adjustments
    suggest_cv_adjustments(cv_score, lb_score, individual_scores)
    
    # Step 6: Future predictions
    if model is not None and len(history) >= 5:
        print("\n=== CV Score Targets ===")
        print("To achieve specific LB scores, aim for these CV scores:")
        
        target_lbs = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10]
        for target_lb in target_lbs:
            # Inverse prediction: CV = (LB - intercept) / coef
            if model.coef_[0] != 0:
                target_cv = (target_lb - model.intercept_) / model.coef_[0]
                if target_cv > 0:
                    print(f"  LB {target_lb:.2f} → Need CV ~{target_cv:.4f}")
    
    # Step 7: Next steps
    print("\n=== Next Steps ===")
    best_lb = history['lb_score'].min()
    if lb_score < best_lb:
        print(f"🎉 New best score! Previous best was {best_lb:.4f}")
    else:
        print(f"Current best LB: {best_lb:.4f}")
        if model is not None:
            cv_needed = (best_lb * 0.95 - model.intercept_) / model.coef_[0] if model.coef_[0] != 0 else 0
            if cv_needed > 0:
                print(f"To beat it by 5%, aim for CV < {cv_needed:.4f}")

if __name__ == "__main__":
    main()