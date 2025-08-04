# Feature List for NeurIPS Polymer Prediction Model

This document lists all 52 features currently used in the polymer prediction model, along with their average importance scores across all targets.

## Overview

- **Total Features**: 52
- **Feature Categories**: Molecular structure, composition, physical properties, and topological descriptors
- **Importance Metric**: Average feature importance from LightGBM models across all 5 targets (Tg, FFV, Tc, Density, Rg)

## Features by Importance

### Top 10 Most Important Features

1. **avg_bond_length** (568.0) - Average bond length in Angstroms calculated from bond types
2. **rg_estimate** (409.0) - Radius of gyration estimate using formula: Rg = sqrt((N × b²) / 6)
3. **density_estimate** (379.2) - Estimated density in g/cm³ from molecular weight and Van der Waals volume
4. **main_branch_atom_ratio** (290.0) - Ratio of main branch atoms to total heavy atoms
5. **chain_length_estimate** (282.8) - Estimated maximum chain length from SMILES segments
6. **heteroatom_ratio** (273.6) - Ratio of heteroatoms (non-C/H) to total heavy atoms
7. **length** (266.4) - Total length of SMILES string
8. **molecular_weight** (248.4) - Estimated molecular weight in g/mol
9. **aromatic_ratio** (226.6) - Ratio of aromatic atoms to total length
10. **vdw_volume** (217.4) - Van der Waals volume in Å³

### Structural Features (11-20)

11. **molecular_complexity** (131.0) - Sum of rings, branches, and chiral centers
12. **num_branches** (119.0) - Number of branches (parentheses) in structure
13. **num_rings** (111.8) - Number of ring structures
14. **num_C** (102.6) - Number of non-aromatic carbon atoms
15. **num_double_bonds** (89.0) - Number of explicit double bonds (=)
16. **main_branch_atoms** (88.8) - Number of atoms in the main polymer backbone
17. **heteroatom_count** (83.8) - Total count of non-C/H atoms
18. **heavy_atom_count** (82.6) - Total count of non-hydrogen atoms
19. **ffv_estimate** (62.6) - Fractional free volume estimate
20. **num_aromatic_atoms** (60.0) - Total count of aromatic atoms

### Functional Group Features (21-30)

21. **has_amine** (48.2) - Binary indicator for presence of amine group
22. **num_single_bonds** (38.6) - Number of explicit single bonds (-)
23. **has_phenyl** (37.0) - Binary indicator for phenyl ring pattern
24. **has_ester** (34.2) - Binary indicator for ester group
25. **flexibility_score** (31.4) - Rotatable bonds normalized by heavy atom count
26. **has_carbonyl** (26.0) - Binary indicator for carbonyl group (C=O)
27. **has_amide** (25.6) - Binary indicator for amide group
28. **has_ether** (22.6) - Binary indicator for ether linkage
29. **num_triple_bonds** (18.8) - Number of triple bonds (#)
30. **has_methyl** (18.2) - Binary indicator for methyl groups

### Atom Count Features (31-40)

31. **num_S** (17.4) - Number of sulfur atoms
32. **num_n** (13.0) - Number of aromatic nitrogen atoms
33. **backbone_bonds** (7.2) - Number of bonds in the main backbone
34. **num_chiral_centers** (5.2) - Number of chiral centers (@)
35. **new_sim** (4.4) - Binary indicator for main vs supplementary dataset
36. **num_Cl** (3.8) - Number of chlorine atoms
37. **rotatable_bond_estimate** (3.4) - Estimated rotatable bonds
38. **has_sulfone** (3.2) - Binary indicator for sulfone group
39. **has_cyclohexyl** (2.6) - Binary indicator for cyclohexyl group
40. **num_F** (2.4) - Number of fluorine atoms

### Low Importance Features (41-52)

41. **has_bridge** (1.2) - Binary indicator for bridged structures
42. **has_fused_rings** (1.0) - Binary indicator for fused ring systems
43. **has_spiro** (0.2) - Binary indicator for spiro centers
44. **num_aromatic_bonds** (0.0) - Number of aromatic bonds (:)
45. **num_Br** (0.0) - Number of bromine atoms
46. **num_c** (0.0) - Number of aromatic carbon atoms
47. **num_o** (0.0) - Number of aromatic oxygen atoms
48. **num_O** (0.0) - Number of non-aromatic oxygen atoms
49. **num_N** (0.0) - Number of non-aromatic nitrogen atoms
50. **num_s** (0.0) - Number of aromatic sulfur atoms
51. **num_I** (0.0) - Number of iodine atoms
52. **num_P** (0.0) - Number of phosphorus atoms

## Target-Specific Feature Selection

The model uses target-specific feature selection for atom count features:

- **Tg**: Uses num_C and num_n
- **FFV**: Uses num_S and num_n
- **Tc**: Uses num_C and num_S
- **Density**: Uses num_Cl and num_Br
- **Rg**: Uses num_F and num_Cl

All other features are used for all targets.

## Feature Engineering Timeline

1. **Basic molecular features**: Atom counts, bond counts, structural patterns
2. **Molecular weight**: Added for physics-based Tg prediction
3. **Van der Waals volume and density**: Added for better physical property estimation
4. **Main branch analysis**: Added to capture polymer backbone characteristics
5. **FFV estimation**: Added using density and volume calculations
6. **Backbone bonds**: Added to count bonds in main polymer chain
7. **Average bond length**: Added to capture bond characteristics
8. **Rg estimation**: Added using polymer physics formula combining backbone bonds and bond length

## Notes

- Features with 0.0 importance may still contribute through interactions with other features
- The importance scores are averaged across all 5 targets, so a feature might be very important for one target but show lower average importance
- Binary features (has_*) use 1 for presence and 0 for absence
- All continuous features are calculated with appropriate precision (typically 3 decimal places)