# Utility Scripts

## diagnostics/
Tools for debugging and understanding model performance:
- `diagnose_cv_lb_gap.py` - Analyze why CV and LB scores differ
- `diagnose_metric.py` - Debug metric calculation issues

## tracking/
Tools for tracking CV/LB correlation:
- `sync_cv_lb.py` - Main tool: Run this after each submission
- `track_cv_lb.py` - Manual tracking of CV/LB scores
- `estimate_lb_score.py` - Predict LB score from CV
- `pre_submission_check.py` - Check if model is worth submitting
- `check_submissions.py` - Analyze submission history
- `submit_and_track.py` - Automated submission (not working for code competitions)

### Quick Usage:
```bash
# After getting LB score from Kaggle:
python utils/tracking/sync_cv_lb.py 0.158 "description"
```