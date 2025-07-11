# Quick Reference Guide

## 🎯 Main Workflow

```bash
# 1. Check status
python status.py

# 2. Generate submission
python model.py

# 3. Evaluate locally
python cross_validation.py

# 4. Copy model.py to Kaggle notebook and submit

# 5. Track results (after getting LB score)
python sync_cv_lb.py 0.158 "description"
```

## 📊 Score Targets

Based on separate models approach (holdout/LB ratio ~1.3x):
- **LB 0.10** → Need holdout < 0.077
- **LB 0.09** → Need holdout < 0.069  
- **LB 0.08** → Need holdout < 0.062
- **LB 0.07** → Need holdout < 0.054

## 🔑 Key Files

- `model.py` - Generates submission.csv for Kaggle
- `cross_validation.py` - Local evaluation (train/val/test/holdout)
- `sync_cv_lb.py` - Track CV vs LB correlation
- `status.py` - Quick project status check

## 📁 Data Locations

- Training data: `data/raw/train.csv`
- Test data: `data/raw/test.csv`
- Submission: `output/submission.csv`
- Tracking: `cv_lb_tracking.csv`

## 🐛 Debugging

If scores seem wrong:
- Check `utils/diagnostics/diagnose_cv_lb_gap.py`
- Check `utils/diagnostics/diagnose_metric.py`

## 📈 Current Best

- Approach: Separate Ridge models per target
- Local Holdout: 0.0616 (better LB predictor)
- Public LB: 0.081 ✨