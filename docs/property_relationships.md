# Polymer Property Relationships

This document explores the relationships and interdependencies between the five target polymer properties in our dataset.

## Missing Value Patterns

Our analysis reveals strong correlations in missing value patterns:

- **Density and Rg**: Almost always missing or present together
- **FFV**: Most commonly available property (88.2% of samples)
- **Tg and Tc**: Rarely co-occur (only 0.4% of samples have both)

### Property Availability Statistics
- Tg: 511 samples (6.4%)
- FFV: 7030 samples (88.2%)
- Tc: 737 samples (9.2%)
- Density: 613 samples (7.7%)
- Rg: 614 samples (7.7%)

## Property Interdependencies

### 1. Tg-FFV Relationship
- At Tg, FFV ≈ 0.025 (2.5%) according to WLF (Williams-Landel-Ferry) theory
- This represents the critical free volume where polymer chains transition from glassy to rubbery state
- Higher Tg generally correlates with lower FFV at room temperature

### 2. Density-Crystallinity Relationship
- Higher crystallinity correlates with higher density
- Crystalline regions pack more efficiently than amorphous regions
- Density can be used to estimate crystallinity percentage:
  - % Crystallinity = [(ρ - ρa)/(ρc - ρa)] × 100

### 3. Rg-MW Relationship
- Follows power law: Rg ∝ Mw^ν
- Scaling exponent ν depends on solvent quality:
  - ν = 0.5 for theta solvent (random coil)
  - ν = 0.6 for good solvent (expanded coil)
  - ν = 0.33 for poor solvent (collapsed globule)
  - ν = 1.0 for rigid rod

### 4. FFV-Permeability Relationship
- Higher FFV typically leads to higher gas permeability
- Critical for membrane applications
- Trade-off exists between permeability and selectivity

### 5. Tg-Tc Exclusivity
- Most polymers have either Tg or Tc, rarely both
- Indicates distinct polymer types:
  - Amorphous: Tg only
  - Crystalline: Tc only
  - Semi-crystalline: Both (rare in our dataset)

## Correlations for Modeling

### Strong Positive Correlations
1. **Density ↔ Rg**: Missing value patterns strongly correlated
2. **Chain stiffness → Higher Tg & Lower FFV**
3. **Crystallinity → Higher Density & Higher Tc**

### Inverse Relationships
1. **FFV ↔ Density**: Higher free volume means lower density
2. **Flexibility ↔ Tg**: More flexible chains have lower Tg
3. **Branching ↔ Crystallinity**: More branches reduce crystallinity

## Implications for Multi-Target Prediction

### Challenge: Sparse Target Matrix
- Most samples have only 1 out of 5 properties measured
- Need to leverage property relationships for prediction

### Strategies
1. **Transfer Learning**: Use abundant FFV data to inform other predictions
2. **Multi-Task Learning**: Share representations between related properties
3. **Physics-Informed Models**: Incorporate known relationships as constraints
4. **Imputation**: Use property correlations to estimate missing values

### Property Groupings for Modeling
1. **Thermal Properties**: Tg, Tc
2. **Structural Properties**: Density, FFV
3. **Solution Properties**: Rg

## Data Quality Observations

### Unusual Patterns
1. **Tc values**: Unusually low (0.05-0.52 K) and tightly clustered
   - May indicate normalized or transformed data
   - Original values might be in different units

2. **Density-Rg coupling**: Almost perfect correlation in missingness
   - Suggests these are measured together experimentally
   - May come from same characterization technique

3. **FFV abundance**: 88% availability vs <10% for others
   - FFV might be calculated rather than measured
   - Could serve as anchor for predicting other properties

## Recommendations for Feature Engineering

1. **Create interaction features** between properties when available
2. **Use FFV as base feature** due to high availability
3. **Engineer crystallinity indicators** from structure
4. **Calculate theoretical bounds** for properties based on structure
5. **Leverage missing patterns** as informative features