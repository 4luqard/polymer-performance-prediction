#!/usr/bin/env python3
"""
Submit to Kaggle and track CV vs Leaderboard correlation
"""

import subprocess
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add .local/bin to PATH for kaggle command
os.environ['PATH'] = os.environ.get('PATH', '') + ':' + os.path.expanduser('~/.local/bin')

COMPETITION_NAME = "neurips-open-polymer-prediction-2025"
SUBMISSION_MESSAGE_PREFIX = "CV_"

def run_cross_validation():
    """Run cross-validation and extract the score"""
    print("Running cross-validation...")
    result = subprocess.run([sys.executable, "cross_validation.py"], 
                          capture_output=True, text=True)
    
    # Parse CV score from output
    cv_score = None
    for line in result.stdout.split('\n'):
        if "Overall Competition Metric (wMAE):" in line:
            try:
                cv_score = float(line.split(':')[1].split('(')[0].strip())
                break
            except:
                pass
    
    if cv_score is None:
        print("Could not extract CV score from output")
        print("Output:", result.stdout[-500:])
        return None
        
    return cv_score

def generate_submission():
    """Generate a new submission file"""
    print("Generating submission...")
    result = subprocess.run([sys.executable, "model.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error generating submission:", result.stderr)
        return False
    
    # Check if submission file exists
    if not os.path.exists("output/submission.csv"):
        print("Submission file not found")
        return False
        
    return True

def submit_to_kaggle(cv_score):
    """Submit to Kaggle with CV score in message"""
    submission_message = f"{SUBMISSION_MESSAGE_PREFIX}{cv_score:.4f}"
    
    print(f"Submitting to Kaggle with message: {submission_message}")
    
    cmd = [
        "kaggle", "competitions", "submit",
        "-f", "output/submission.csv",
        "-m", submission_message,
        COMPETITION_NAME
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error submitting:", result.stderr)
        print("Stdout:", result.stdout)
        return False
    
    print("Submission successful!")
    print(result.stdout)
    return True

def get_latest_submission_score(wait_time=60, max_attempts=10):
    """Get the score of the latest submission"""
    print(f"Waiting {wait_time} seconds for score to appear...")
    time.sleep(wait_time)
    
    for attempt in range(max_attempts):
        cmd = [
            "kaggle", "competitions", "submissions",
            COMPETITION_NAME,
            "-v"  # CSV format
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error getting submissions:", result.stderr)
            return None
        
        # Parse CSV output
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(result.stdout))
            if len(df) > 0:
                latest = df.iloc[0]
                
                # Check if this is our submission (by message prefix)
                if str(latest.get('description', '')).startswith(SUBMISSION_MESSAGE_PREFIX):
                    if pd.notna(latest.get('publicScore')):
                        return float(latest['publicScore'])
                    else:
                        print(f"Attempt {attempt + 1}: Score not yet available...")
                        time.sleep(30)
                else:
                    print(f"Latest submission is not ours: {latest.get('description', '')}")
                    return None
        except Exception as e:
            print(f"Error parsing submissions: {e}")
            return None
    
    print("Could not get score after maximum attempts")
    return None

def load_tracking_history():
    """Load existing tracking history"""
    history_file = "cv_lb_tracking.csv"
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    else:
        return pd.DataFrame(columns=['timestamp', 'cv_score', 'lb_score', 'difference', 'ratio'])

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
    print(df.tail(5)[['timestamp', 'cv_score', 'lb_score', 'difference']].to_string(index=False))
    
    # Suggest adjustment factor
    if len(df) >= 3:
        # Use linear regression to find best mapping
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(df[['cv_score']], df['lb_score'])
        
        print(f"\nSuggested LB estimate: LB ≈ {lr.coef_[0]:.4f} * CV + {lr.intercept_:.4f}")

def main():
    """Main workflow"""
    print("=== CV to Leaderboard Synchronization ===")
    
    # Step 1: Run CV
    cv_score = run_cross_validation()
    if cv_score is None:
        print("Failed to get CV score")
        return
    
    print(f"\nCV Score: {cv_score:.4f}")
    
    # Step 2: Generate submission
    if not generate_submission():
        print("Failed to generate submission")
        return
    
    # Step 3: Submit to Kaggle
    if not submit_to_kaggle(cv_score):
        print("Failed to submit to Kaggle")
        return
    
    # Step 4: Get leaderboard score
    lb_score = get_latest_submission_score()
    if lb_score is None:
        print("Failed to get leaderboard score")
        return
    
    print(f"\nLeaderboard Score: {lb_score:.4f}")
    
    # Step 5: Track results
    history = load_tracking_history()
    
    new_entry = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cv_score': cv_score,
        'lb_score': lb_score,
        'difference': cv_score - lb_score,
        'ratio': cv_score / lb_score if lb_score > 0 else np.nan
    }])
    
    history = pd.concat([history, new_entry], ignore_index=True)
    save_tracking_history(history)
    
    # Step 6: Analyze correlation
    analyze_correlation(history)
    
    print("\n=== Tracking complete ===")

if __name__ == "__main__":
    main()