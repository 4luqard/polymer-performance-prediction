#!/usr/bin/env python3
"""
Quick status check for the project
"""

import os
import pandas as pd

print("🎯 NeurIPS Polymer Prediction Status\n")

# Check if we have tracking data
if os.path.exists("cv_lb_tracking.csv"):
    df = pd.read_csv("cv_lb_tracking.csv")
    print(f"📊 Submissions tracked: {len(df)}")
    if len(df) > 0:
        best = df.loc[df['lb_score'].idxmin()]
        latest = df.iloc[-1]
        print(f"🏆 Best LB: {best['lb_score']:.4f} (CV: {best['cv_score']:.4f})")
        print(f"📍 Latest: LB {latest['lb_score']:.4f} (CV: {latest['cv_score']:.4f})")
        
        if len(df) >= 2:
            ratio = df['lb_score'].mean() / df['cv_score'].mean()
            print(f"📈 Avg CV/LB ratio: {ratio:.1f}x")
else:
    print("❌ No submissions tracked yet")
    print("   Run: python sync_cv_lb.py <LB_SCORE> after submission")

print("\n🚀 Next steps:")
print("1. python model.py          # Generate submission")
print("2. python cross_validation.py  # Check local score")
print("3. Submit to Kaggle notebook")
print("4. python sync_cv_lb.py <LB>   # Track results")