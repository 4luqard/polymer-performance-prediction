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

Based on CV/LB ratio of 13.2x:
- **LB 0.15** → Need CV < 0.0114
- **LB 0.14** → Need CV < 0.0106  
- **LB 0.13** → Need CV < 0.0099
- **LB 0.10** → Need CV < 0.0076

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

- Local CV: 0.0122
- Local Holdout: 0.0118
- Public LB: 0.158