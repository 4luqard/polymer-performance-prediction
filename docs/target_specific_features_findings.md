# Target-Specific Features Investigation

## Background (2025-07-26)
Following the atom count feature analysis, we investigated using only the top 2 most important atom count features for each target, rather than all atom features.

## Analysis Performed

### Top 2 Atom Features per Target
Based on Ridge regression coefficient importance:

- **Tg**: `num_C` (19.65), `num_n` (16.31)
- **FFV**: `num_S` (0.0070), `num_n` (0.0056) 
- **Tc**: `num_C` (0.0163), `num_S` (0.0128)
- **Density**: `num_Cl` (0.0335), `num_Br` (0.0229)
- **Rg**: `num_F` (0.6757), `num_Cl` (0.4611)

### Implementation
Modified `model.py` to support target-specific feature selection:
- Each target model uses all non-atom features + its top 2 atom features
- This reduces features from 42 to approximately 31 per target

### Cross-Validation Results

**Full atom features (baseline)**:
- CV Score: 0.0670 (±0.0035)
- Uses all 42 features for all targets

**Target-specific atom features**:
- CV Score: 0.0680 (±0.0034)
- Fold scores: [0.0625, 0.0711, 0.0657, 0.0694, 0.0715]
- Performance decreased by 0.0010

## Conclusion

Using target-specific atom features does NOT improve performance:
- The model performs better when all targets have access to all atom features
- The slight increase in CV score (0.0010) indicates worse performance
- The added complexity is not justified by the results

## Recommendation

Keep the current approach with all atom features for all targets. The model benefits from having access to the full set of atom counts, even if some are less important for specific targets.