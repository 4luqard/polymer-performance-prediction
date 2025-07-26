# Feature Analysis Findings

## Overfitting Investigation (2025-07-26)

### Background
Concern was raised about potential overfitting from certain features, specifically:
- `has_polymer_end`
- Atom count features (`num_C`, `num_O`, `num_N`, etc.)

### Analysis Performed

#### 1. Feature Importance Analysis
Analyzed coefficient-based importance for each target using Ridge regression:

**Key Findings:**
- `has_polymer_end` showed **zero importance** across all targets (rank 22/43)
- Atom count features showed varying importance:
  - **Tg**: `num_C` is highly important (rank 2/43, importance=19.65)
  - **FFV**: Lower importance but still contributing
  - **Density**: `num_Cl` (rank 3) and `num_Br` (rank 6) are important
  - **Rg**: `num_F` and `num_Cl` moderately important

#### 2. Cross-Validation Testing

**Test 1: Removing all atom counts and has_polymer_end**
- Full features CV score: 0.0676 (±0.0021)
- Reduced features CV score: 0.0706 (±0.0020)
- **Result**: Model performs worse without atom counts (-0.0030)

**Test 2: Removing only has_polymer_end**
- With has_polymer_end: 0.0670 (±0.0035)
- Without has_polymer_end: 0.0670 (±0.0035)
- **Result**: No difference in performance

### Conclusions

1. **Atom count features are valuable** - They improve model performance and should be kept
2. **has_polymer_end is useless** - Zero importance and no impact on performance

### Actions Taken

1. Removed `has_polymer_end` feature from model.py
2. Kept all atom count features as they improve predictions
3. Model now uses 42 features instead of 43

### Final Model Performance
- CV Score: 0.0670 (±0.0035)
- Public LB Score: 0.081 (best submission)