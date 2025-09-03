# NeurIPS Open Polymer Prediction 2025 - Project Documentation

## Project Overview
This project tackles the NeurIPS 2025 Open Polymer Prediction challenge, aiming to predict polymer properties directly from their chemical structure (SMILES representation).

## Competition Task
- **Goal**: Predict 5 polymer properties from SMILES molecular representations
- **Target Properties**:
  - Tg: Glass transition temperature
  - FFV: Fractional free volume  
  - Tc: Crystallization temperature
  - Density: Material density
  - Rg: Radius of gyration
- **Evaluation**: Weighted Mean Absolute Error (wMAE) with property-specific normalization
- **Challenge**: Training data has ~91% samples with only 1 target value (significant missing data)

## Codebase Architecture

### Core Modules
- `model.py`: Main model implementation with LightGBM ensemble
- `data_processing.py`: Feature engineering pipeline (52 molecular descriptors)
- `config.py`: Environment-aware configuration management
- `transformer_model.py`: Neural network alternatives with SMILES tokenization
- `cv.py`: Cross-validation utilities
- `residual_analysis.py`: Model performance diagnostics

### Directory Structure
```
neurips-open-polymer-prediction-2025/
├── data/               # Competition datasets
├── src/                # Core utilities and helpers
├── tests/              # Comprehensive test suite
├── scripts/            # Automation and deployment
├── notebooks/          # Experimental analysis
├── output/             # Model outputs and predictions
└── residual_analysis/  # Performance diagnostics
```

## Code Style Guidelines

### Naming Conventions
- **Functions/Variables**: snake_case (e.g., `extract_molecular_features`)
- **Constants**: UPPERCASE (e.g., `TARGET_FEATURES`)
- **Classes**: CamelCase (e.g., `SMILESTokenizer`)
- **Modules**: lowercase with underscores

### Organization Principles
- Single responsibility per module
- Clear separation of data processing, modeling, and evaluation
- Environment-aware design (Kaggle vs local)
- Comprehensive error handling and logging

## Technical Implementation

### Feature Engineering
52 engineered features from SMILES including:
- Structural features (bonds, rings, branching)
- Compositional features (atom counts, heteroatom ratios)
- Physical properties (molecular weight, density estimates)
- Polymer-specific features (backbone bonds, main chain atoms)

### Model Strategy
- **Primary Model**: LightGBM with target-specific configurations
- **Architecture**: Separate models for each of 5 targets
- **Feature Selection**: Target-specific feature importance analysis
- **Validation**: Custom CV with competition metric

### Key Libraries
- **ML Stack**: LightGBM, scikit-learn, TensorFlow/Keras
- **Data Processing**: pandas, numpy, RDKit
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, pathlib, warnings

## Performance Metrics
- **Current CV Score**: 0.0541 (competition wMAE)
- **Model Comparison**: LightGBM > Ridge > Transformer
- **Data Efficiency**: Effective handling of 91% missing values

## Running the Code

### Environment Setup
```bash
# Local execution
python3 model.py

# Cross-validation only
python3 model.py --cv-only

# Kaggle notebook execution
# Code auto-detects Kaggle environment
```

### Key Requirements
- Must run offline (Kaggle constraint)
- CPU-only TensorFlow for reproducibility
- Deterministic seed management
- Environment-aware path handling

## Testing
Comprehensive test suite using pytest:
- Data integration tests
- Model functionality tests
- Feature engineering validation
- Cross-validation workflows

Run tests with: `pytest tests/`

## Git Configuration
- Single-sentence commit messages
- No co-author attribution
- Atomic commits (one change per commit)
- Author: 4luqard (turhan.m.tas@gmail.com)

## Development Guidelines
1. Maintain modular architecture
2. Document significant changes
3. Ensure Kaggle compatibility
4. Run tests before commits
5. Follow existing code patterns
6. Preserve reproducibility with seed management

## Current Focus Areas
- Feature engineering refinement
- Missing value imputation strategies
- Model ensemble optimization
- Cross-validation improvement
- Submission file generation for Kaggle

This project demonstrates strong software engineering practices with a focus on reproducibility, maintainability, and performance optimization for polymer property prediction.