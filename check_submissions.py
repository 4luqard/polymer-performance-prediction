#!/usr/bin/env python3
"""
Check submission history and analyze CV vs LB correlation
"""

import subprocess
import json
import pandas as pd
import numpy as np
import os

# Add .local/bin to PATH for kaggle command
os.environ['PATH'] = os.environ.get('PATH', '') + ':' + os.path.expanduser('~/.local/bin')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

COMPETITION_NAME = "neurips-open-polymer-prediction-2025"

def get_all_submissions():
    """Get all submissions from Kaggle"""
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
        # Convert to list of dicts for compatibility
        submissions = df.to_dict('records')
        return submissions
    except Exception as e:
        print(f"Error parsing submissions CSV: {e}")
        print("Output:", result.stdout[:500])
        return None

def parse_cv_score_from_description(description):
    """Extract CV score from submission description"""
    if description and description.startswith("CV_"):
        try:
            return float(description.split("CV_")[1].split()[0])
        except:
            pass
    return None

def analyze_submissions():
    """Analyze all submissions and their scores"""
    submissions = get_all_submissions()
    
    if not submissions:
        print("No submissions found")
        return
    
    print(f"Total submissions: {len(submissions)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(submissions)
    
    # Parse CV scores from descriptions
    df['cv_score'] = df['description'].apply(parse_cv_score_from_description)
    df['lb_score'] = pd.to_numeric(df['publicScore'], errors='coerce')
    
    # Filter to only submissions with CV scores
    cv_submissions = df[df['cv_score'].notna() & df['lb_score'].notna()].copy()
    
    if len(cv_submissions) == 0:
        print("\nNo submissions found with CV scores in description")
        print("\nAll submissions:")
        print(df[['date', 'description', 'publicScore', 'status']].head(10))
        return
    
    print(f"\nSubmissions with CV tracking: {len(cv_submissions)}")
    
    # Calculate differences
    cv_submissions['difference'] = cv_submissions['cv_score'] - cv_submissions['lb_score']
    cv_submissions['ratio'] = cv_submissions['cv_score'] / cv_submissions['lb_score']
    
    # Analysis
    print("\n=== CV vs Leaderboard Analysis ===")
    
    if len(cv_submissions) >= 2:
        correlation = cv_submissions['cv_score'].corr(cv_submissions['lb_score'])
        print(f"Correlation (CV vs LB): {correlation:.4f}")
    
    print(f"Average CV score: {cv_submissions['cv_score'].mean():.4f}")
    print(f"Average LB score: {cv_submissions['lb_score'].mean():.4f}")
    print(f"Average difference (CV - LB): {cv_submissions['difference'].mean():.4f} (+/- {cv_submissions['difference'].std():.4f})")
    print(f"Average ratio (CV / LB): {cv_submissions['ratio'].mean():.4f}")
    
    # Show recent tracked submissions
    print("\nRecent CV-tracked submissions:")
    display_cols = ['date', 'cv_score', 'lb_score', 'difference', 'status']
    print(cv_submissions[display_cols].head(10).to_string(index=False))
    
    # Plot if we have enough data
    if len(cv_submissions) >= 3 and HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(cv_submissions['cv_score'], cv_submissions['lb_score'], alpha=0.6)
        
        # Add diagonal line
        min_score = min(cv_submissions['cv_score'].min(), cv_submissions['lb_score'].min())
        max_score = max(cv_submissions['cv_score'].max(), cv_submissions['lb_score'].max())
        plt.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5)
        
        # Add best fit line
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        X = cv_submissions[['cv_score']].values
        y = cv_submissions['lb_score'].values
        lr.fit(X, y)
        plt.plot(cv_submissions['cv_score'], lr.predict(X), 'g-', 
                label=f'LB = {lr.coef_[0]:.3f}*CV + {lr.intercept_:.3f}')
        
        plt.xlabel('CV Score')
        plt.ylabel('Leaderboard Score')
        plt.title('CV vs Leaderboard Score')
        plt.legend()
        
        # Time series
        plt.subplot(1, 2, 2)
        cv_submissions['date'] = pd.to_datetime(cv_submissions['date'])
        cv_submissions_sorted = cv_submissions.sort_values('date')
        
        plt.plot(range(len(cv_submissions_sorted)), cv_submissions_sorted['cv_score'], 
                'b-o', label='CV Score', markersize=6)
        plt.plot(range(len(cv_submissions_sorted)), cv_submissions_sorted['lb_score'], 
                'r-s', label='LB Score', markersize=6)
        
        plt.xlabel('Submission #')
        plt.ylabel('Score')
        plt.title('Score Progress Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cv_lb_analysis.png', dpi=150)
        print("\nPlot saved to cv_lb_analysis.png")
    elif len(cv_submissions) >= 3:
        # Still do the linear regression analysis without plotting
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        X = cv_submissions[['cv_score']].values
        y = cv_submissions['lb_score'].values
        lr.fit(X, y)
        print(f"\nLinear regression: LB = {lr.coef_[0]:.3f}*CV + {lr.intercept_:.3f}")
    
    # Best submission info
    print("\n=== Best Submissions ===")
    best_lb = df.loc[df['lb_score'].idxmin()]
    print(f"Best LB score: {best_lb['lb_score']:.4f} ({best_lb['description']}) on {best_lb['date']}")
    
    if len(cv_submissions) > 0:
        best_cv_tracked = cv_submissions.loc[cv_submissions['lb_score'].idxmin()]
        print(f"Best CV-tracked LB score: {best_cv_tracked['lb_score']:.4f} (CV: {best_cv_tracked['cv_score']:.4f}) on {best_cv_tracked['date']}")

def main():
    """Main function"""
    print("=== Kaggle Submission History Analysis ===")
    analyze_submissions()

if __name__ == "__main__":
    main()