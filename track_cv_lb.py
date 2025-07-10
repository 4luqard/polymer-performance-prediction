#!/usr/bin/env python3
"""
Track CV vs LB scores by manually entering them
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

def load_tracking_history():
    """Load existing tracking history"""
    history_file = "cv_lb_tracking.csv"
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    else:
        return pd.DataFrame(columns=['timestamp', 'cv_score', 'lb_score', 'difference', 'ratio', 'notes'])

def save_tracking_history(df):
    """Save tracking history"""
    df.to_csv("cv_lb_tracking.csv", index=False)

def analyze_correlation(df):
    """Analyze CV vs LB correlation"""
    if len(df) < 2:
        print("Not enough data points for correlation analysis")
        return
    
    print("\n=== CV vs Leaderboard Analysis ===")
    print(f"Number of submissions tracked: {len(df)}")
    
    # Correlation
    correlation = df['cv_score'].corr(df['lb_score'])
    print(f"Correlation (CV vs LB): {correlation:.4f}")
    
    # Average difference
    avg_diff = df['difference'].mean()
    std_diff = df['difference'].std()
    print(f"Average difference (CV - LB): {avg_diff:.4f} (+/- {std_diff:.4f})")
    
    # Average ratio
    avg_ratio = df['ratio'].mean()
    print(f"Average ratio (CV / LB): {avg_ratio:.4f}")
    
    # Show recent history
    print("\nRecent submissions:")
    display_cols = ['timestamp', 'cv_score', 'lb_score', 'difference', 'notes']
    print(df.tail(10)[display_cols].to_string(index=False))
    
    # Suggest adjustment factor
    if len(df) >= 3:
        # Use linear regression to find best mapping
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(df[['cv_score']], df['lb_score'])
        
        print(f"\nSuggested LB estimate: LB ≈ {lr.coef_[0]:.4f} * CV + {lr.intercept_:.4f}")
        
        # Predict current CV
        if '--predict' in sys.argv:
            cv_score = float(sys.argv[sys.argv.index('--predict') + 1])
            predicted_lb = lr.predict([[cv_score]])[0]
            print(f"\nPredicted LB for CV={cv_score:.4f}: {predicted_lb:.4f}")

def add_entry(cv_score, lb_score, notes=""):
    """Add a new CV/LB entry"""
    history = load_tracking_history()
    
    new_entry = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cv_score': cv_score,
        'lb_score': lb_score,
        'difference': cv_score - lb_score,
        'ratio': cv_score / lb_score if lb_score > 0 else np.nan,
        'notes': notes
    }])
    
    history = pd.concat([history, new_entry], ignore_index=True)
    save_tracking_history(history)
    
    print(f"Added entry: CV={cv_score:.4f}, LB={lb_score:.4f}, Diff={cv_score-lb_score:.4f}")
    
    analyze_correlation(history)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python track_cv_lb.py add <cv_score> <lb_score> [notes]")
        print("  python track_cv_lb.py analyze")
        print("  python track_cv_lb.py analyze --predict <cv_score>")
        print("  python track_cv_lb.py show")
        return
    
    command = sys.argv[1]
    
    if command == "add":
        if len(sys.argv) < 4:
            print("Error: Need CV and LB scores")
            return
        cv_score = float(sys.argv[2])
        lb_score = float(sys.argv[3])
        notes = " ".join(sys.argv[4:]) if len(sys.argv) > 4 else ""
        add_entry(cv_score, lb_score, notes)
    
    elif command == "analyze":
        history = load_tracking_history()
        if len(history) > 0:
            analyze_correlation(history)
        else:
            print("No tracking history found")
    
    elif command == "show":
        history = load_tracking_history()
        if len(history) > 0:
            print(history.to_string(index=False))
        else:
            print("No tracking history found")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()