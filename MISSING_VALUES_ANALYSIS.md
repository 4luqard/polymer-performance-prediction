# Missing Values Analysis - NeurIPS Open Polymer Prediction 2025

## Executive Summary

**Critical Finding**: No training sample has all 5 targets (Tg, FFV, Tc, Density, Rg) available. This is by design and reflects real experimental limitations, not random missing data.

## Key Statistics

- **Training samples**: 7,973
- **Test samples**: 3
- **Target properties**: Tg, FFV, Tc, Density, Rg

### Target Availability
| Target  | Available | Missing | Availability % |
|---------|-----------|---------|----------------|
| FFV     | 7,030     | 943     | 88.2%         |
| Tc      | 737       | 7,236   | 9.2%          |
| Density | 613       | 7,360   | 7.7%          |
| Rg      | 614       | 7,359   | 7.7%          |
| Tg      | 511       | 7,462   | 6.4%          |

## Missing Value Patterns

### Top 5 Most Common Patterns
1. **01000** (83.8%): Only FFV available - 6,681 samples
2. **10000** (5.9%): Only Tg available - 473 samples  
3. **00111** (3.7%): Tc+Density+Rg available - 292 samples
4. **01111** (2.8%): FFV+Tc+Density+Rg available - 221 samples
5. **00100** (1.4%): Only Tc available - 110 samples

**Total unique patterns**: 16

## Target Correlations & Co-occurrence

### Strong Correlations
- **Density ↔ Rg**: When Density is available, Rg is available 99.5% of the time
- **Tc ↔ Density**: When Tc is available, Density is available 72% of the time
- **Tc ↔ Rg**: When Tc is available, Rg is available 87% of the time

### Weak Correlations
- **FFV ↔ Tg**: Almost never appear together (0.1% overlap)
- **FFV ↔ Others**: FFV rarely co-occurs with Tc/Density/Rg (3-4%)

## Chemical Structure Analysis

### By Functional Group
- **Amine-containing**: 4,766 samples (91.9% have FFV)
- **Fluorine-containing**: 818 samples (93.6% have FFV)
- **Ether-containing**: 1,073 samples (85.6% have FFV)
- **Aromatic**: 419 samples (89.5% have FFV)

## Hypothesis: Why Values Are Missing

1. **Different Experimental Sources**: Dataset aggregated from multiple research papers/labs
2. **Measurement Technique Specialization**: Different labs specialize in different property measurements
3. **Polymer Type Constraints**: Some properties are harder to measure for certain polymer types
4. **Intentional Competition Design**: Multi-target learning with incomplete data

## Modeling Implications

### Key Insights
1. **No Complete Training Examples**: Cannot train on samples with all targets
2. **FFV as Primary Target**: Most data available, could be main prediction anchor
3. **Tc/Density/Rg Cluster**: These properties are measured together, suggesting related experimental techniques
4. **Test Set Challenge**: Must predict ALL targets despite incomplete training

### Recommended Strategies
1. **Multi-target Learning**: Handle missing values explicitly
2. **Separate Models**: Train individual models for each target
3. **Cross-target Features**: Use available targets as features for missing ones
4. **Domain Knowledge**: Leverage polymer physics relationships
5. **Imputation**: Consider advanced imputation techniques

## Competition Metric Considerations

The competition uses weighted Mean Absolute Error (wMAE) with:
- Scaling by property range
- Weighting by inverse square root of valid samples

This metric design confirms that missing values are intentional and models must handle incomplete data effectively.

## Next Steps

1. Implement multi-target learning approach
2. Explore cross-target relationships
3. Consider polymer physics-based features
4. Test imputation strategies
5. Validate on patterns seen in training data

---
*Analysis completed: 2025-07-18*
*Competition: NeurIPS Open Polymer Prediction 2025*