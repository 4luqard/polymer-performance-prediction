# First-Principles Approach to Polymer Property Prediction

This document outlines how to predict polymer properties (Tg, FFV, Tc, Density, Rg) using first-principles chemistry and physics, without relying on machine learning models.

## Table of Contents
1. [Glass Transition Temperature (Tg)](#glass-transition-temperature-tg)
2. [Fractional Free Volume (FFV)](#fractional-free-volume-ffv)
3. [Crystallization Temperature (Tc)](#crystallization-temperature-tc)
4. [Density](#density)
5. [Radius of Gyration (Rg)](#radius-of-gyration-rg)
6. [Practical Implementation](#practical-implementation)

## Glass Transition Temperature (Tg)

### Definition
The temperature at which a polymer transitions from a hard, glassy state to a soft, rubbery state.

### First-Principles Prediction

#### 1. **Fox Equation** (for copolymers)
```
1/Tg = Σ(wi/Tgi)
```
Where:
- wi = weight fraction of component i
- Tgi = glass transition temperature of homopolymer i

#### 2. **Group Contribution Method (Van Krevelen)**
```
Tg = (Σ Yi × Mi) / (Σ Mi)
```
Where:
- Yi = molar glass transition function for group i
- Mi = molecular weight of group i

#### 3. **Free Volume Theory**
```
Tg = B / (2.303 × fg)
```
Where:
- B = constant (~1000 K)
- fg = fractional free volume at Tg (~0.025)

### Key Factors from SMILES:
- **Chain stiffness**: Aromatic rings (c) increase Tg
- **Side groups**: Bulky groups increase Tg
- **Intermolecular forces**: H-bonds, polar groups increase Tg
- **Chain symmetry**: Asymmetric structures increase Tg

### Estimation Rules:
- Base aliphatic polymer: Tg ≈ -100 to 0°C
- Add +20-40°C per aromatic ring in backbone
- Add +10-30°C per polar group (C=O, OH, CN)
- Add +5-15°C per bulky side group
- Subtract 10-20°C per flexible ether linkage

## Fractional Free Volume (FFV)

### Definition
The fraction of polymer volume not occupied by polymer molecules.

### First-Principles Calculation

#### 1. **Bondi's Method**
```
FFV = (V - V0) / V = (V - 1.3 × Vw) / V
```
Where:
- V = specific volume (1/density)
- V0 = occupied volume
- Vw = van der Waals volume

#### 2. **Group Contribution Method**
```
Vw = Σ(ni × Vwi)
```
Where:
- ni = number of group i
- Vwi = van der Waals volume of group i

### Van der Waals Volumes (Å³):
- CH3: 33.5
- CH2: 26.7
- CH: 20.0
- C: 13.2
- Aromatic C: 16.5
- O (ether): 11.5
- O (carbonyl): 14.0
- N: 12.0
- F: 10.5
- Cl: 26.0

### Estimation from SMILES:
1. Count each group type
2. Calculate total Vw
3. Estimate density (see below)
4. Calculate FFV using Bondi's equation

## Crystallization Temperature (Tc)

### Definition
Temperature at which polymer chains organize into crystalline structures upon cooling.

### First-Principles Relationships

#### 1. **Hoffman-Weeks Equation**
```
Tm° = Tc / (1 - 1/2γ)
```
Where:
- Tm° = equilibrium melting temperature
- γ = fold surface free energy parameter (~2)

#### 2. **Empirical Relationship**
```
Tc ≈ 0.85 × Tm  (for most polymers)
Tm ≈ Tg + 100°C  (rough estimate)
```

### Crystallinity Factors from SMILES:
- **Regular structure**: Promotes crystallization
- **Linear chains**: Higher Tc
- **Symmetry**: Higher Tc
- **Flexible bonds**: Lower Tc
- **Bulky side groups**: Inhibit crystallization

### Estimation Approach:
1. Assess chain regularity (alternating patterns)
2. Count disrupting elements (branches, bulky groups)
3. Estimate crystallinity tendency (0-1 scale)
4. If crystallinity > 0.3: Tc ≈ 0.85 × (Tg + 100)
5. If crystallinity < 0.3: Tc undefined (amorphous)

## Density

### Definition
Mass per unit volume of the polymer.

### First-Principles Calculation

#### 1. **Group Contribution Method (Van Krevelen)**
```
ρ = (Σ ni × Mi) / (Σ ni × Vi)
```
Where:
- ni = number of group i
- Mi = molecular weight of group i
- Vi = molar volume of group i at 298K

### Molar Volumes at 298K (cm³/mol):
- CH3: 33.5
- CH2: 16.1
- CH: 13.5
- C: 11.0
- Aromatic CH: 13.7
- C=O: 22.0
- O (ether): 10.0
- OH: 14.0
- N: 12.0
- F: 11.0
- Cl: 24.0

#### 2. **Amorphous Density Estimation**
```
ρa = 1.2 × ρvdw
```
Where ρvdw is calculated from van der Waals volumes

### SMILES-Based Estimation:
1. Parse repeating unit structure
2. Count each atomic group
3. Sum molecular weights: M = Σ(ni × Mi)
4. Sum molar volumes: V = Σ(ni × Vi)
5. Calculate density: ρ = M/V

## Radius of Gyration (Rg)

### Definition
Root mean square distance of chain segments from the center of mass.

### First-Principles Models

#### 1. **Freely Jointed Chain**
```
Rg² = (N × b²) / 6
```
Where:
- N = number of backbone bonds
- b = average bond length (~1.5 Å)

#### 2. **Worm-Like Chain Model**
```
Rg² = (L × lp) / 3 × [1 - 3lp/L + 6(lp/L)² - 6(lp/L)³(1 - exp(-L/lp))]
```
Where:
- L = contour length
- lp = persistence length

#### 3. **Kuhn Model**
```
Rg = (N/6)^0.5 × lk
```
Where:
- N = number of Kuhn segments
- lk = Kuhn length

### Persistence Length Estimation:
- Flexible aliphatic: lp ≈ 0.5-1 nm
- Semi-flexible: lp ≈ 1-3 nm
- Rigid aromatic: lp ≈ 3-10 nm

### SMILES-Based Calculation:
1. Count backbone atoms (exclude side groups)
2. Identify rigid segments (aromatic, C=C)
3. Estimate persistence length:
   - Base: 0.7 nm
   - +0.5 nm per aromatic ring
   - +0.3 nm per C=C bond
   - -0.2 nm per ether linkage
4. Calculate Rg using appropriate model

## Practical Implementation

### Step-by-Step Approach:

1. **Parse SMILES Structure**
   - Identify repeating unit
   - Count atomic groups
   - Identify functional groups
   - Assess symmetry and regularity

2. **Calculate Molecular Parameters**
   - Molecular weight of repeat unit
   - Van der Waals volume
   - Molar volume
   - Number of rotatable bonds
   - Aromatic content

3. **Apply Property Equations**
   - **Tg**: Use group contribution + corrections
   - **FFV**: Calculate from Vw and density
   - **Tc**: Assess crystallinity, then estimate
   - **Density**: Group contribution method
   - **Rg**: Chain model based on rigidity

### Example Calculation for Polyethylene (-CH2-CH2-)n:

1. **Structure Analysis**
   - Repeat unit: C2H4
   - MW = 28 g/mol
   - All flexible bonds
   - No polar groups

2. **Property Predictions**
   - **Tg**: Base -100°C + no corrections = -100°C
   - **Density**: ρ = 28 / (2×16.1) = 0.87 g/cm³
   - **FFV**: (1/0.87 - 1.3×53.4/28) / (1/0.87) = 0.18
   - **Tc**: High crystallinity → ~90°C
   - **Rg**: For 100 units: (200×1.5²/6)^0.5 = 8.7 Å

### Validation Ranges:
- Tg: -150 to 400°C
- FFV: 0.1 to 0.4
- Tc: 0 to 350°C (or undefined)
- Density: 0.8 to 2.2 g/cm³
- Rg: 5 to 100 Å (for typical MW)

### Limitations:
- Assumes linear polymers
- Neglects tacticity effects
- Approximates copolymer interactions
- Requires careful group identification
- Less accurate for complex architectures

## References:
1. Van Krevelen, D.W. "Properties of Polymers" (2009)
2. Bicerano, J. "Prediction of Polymer Properties" (2002)
3. Askadskii, A.A. "Computational Materials Science of Polymers" (2003)
4. Miller-Chou, B.A. & Koenig, J.L. "A review of polymer dissolution" (2003)