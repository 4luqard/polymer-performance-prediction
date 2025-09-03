# NeurIPS Open Polymer Prediction 2025 - Project Documentation

## Project Overview
This project tackles the NeurIPS 2025 Open Polymer Prediction challenge, aiming to predict polymer properties directly from their chemical structure (SMILES representation).

## Competition Task
- **Goal**: Predict 5 polymer properties from SMILES molecular representations
- **Target Properties**:
  - Tg: Glass transition temperature (where polymer transitions from hard/glassy to soft/rubbery)
  - FFV: Fractional free volume (packing efficiency)
  - Tc: Thermal conductivity (response to heat)
  - Density: Material density
  - Rg: Radius of gyration (fundamental molecular size)
- **Evaluation**: Weighted Mean Absolute Error (wMAE) with property-specific normalization
- **Challenge**: Training data has ~91% samples with only 1 target value (significant missing data)
- **Ground Truth**: Averaged from multiple runs of molecular dynamics simulation

## Competition Specifics
- **Type**: Code competition (offline execution in Kaggle notebooks)
- **Submission Format**: CSV with predictions for all 5 properties per sample
- **Evaluation Formula**: Includes scale normalization, inverse square-root scaling for rare properties, and weight normalization

## Codebase Architecture

### Core Modules
- `model.py`: Main model implementation with target-specific LightGBM ensemble
- `data_processing.py`: Comprehensive feature engineering pipeline (52 molecular descriptors)
- `config.py`: Environment-aware configuration (Kaggle vs local detection)
- `transformer_model.py`: Neural network alternatives with SMILES tokenization
- `cv.py`: Cross-validation utilities with competition metric
- `residual_analysis.py`: Model performance diagnostics and analysis
- `data_helpers.py`: Data loading and preprocessing utilities

### Directory Structure
```
neurips-open-polymer-prediction-2025/
├── data/               # Competition datasets (train.csv, test.csv, sample_submission.csv)
├── src/                # Core utilities and helper modules
├── tests/              # Comprehensive pytest test suite
├── scripts/            # Automation and deployment scripts
├── notebooks/          # Jupyter notebooks for experimental analysis
├── output/             # Model outputs, predictions, and CV results
├── residual_analysis/  # Performance diagnostics and visualizations
└── venv/              # Python virtual environment
```

## Code Style Guidelines

### Naming Conventions
- **Functions/Variables**: snake_case (e.g., `extract_molecular_features`, `calculate_backbone_bonds`)
- **Constants**: UPPERCASE with underscores (e.g., `TARGET_FEATURES`, `PCA_VARIANCE_THRESHOLD`)
- **Classes**: CamelCase (e.g., `SMILESTokenizer`, `ResidualAnalyzer`, `EnvironmentConfig`)
- **Modules**: lowercase with underscores (e.g., `data_processing.py`, `residual_analysis.py`)

### Code Organization Principles
- Single responsibility per module
- Clear separation of concerns (data processing, modeling, evaluation)
- Environment-aware design with automatic detection
- Comprehensive error handling with warnings suppression
- Extensive use of docstrings for complex functions
- Modular feature engineering with reusable components

## Technical Implementation

### Feature Engineering Pipeline
52 engineered features from SMILES including:
- **Structural Features**: 
  - Bond types (single, double, triple, aromatic)
  - Ring counts and sizes
  - Branching patterns and complexity
- **Compositional Features**:
  - Atom counts (C, N, O, S, F, Cl, etc.)
  - Heteroatom ratios and diversity
  - Element ratios and distributions
- **Physical Properties**:
  - Molecular weight estimates
  - Density approximations
  - Bond length averages
- **Polymer-Specific Features**:
  - Main branch atoms
  - Backbone bonds
  - Polymer end markers
  - Chain complexity metrics

### Modeling Strategy
- **Primary Model**: LightGBM with target-specific hyperparameters
- **Architecture**: Independent models for each of 5 targets
- **Feature Selection**: Target-specific importance analysis and selection
- **Imputation**: SimpleImputer with median strategy
- **Scaling**: StandardScaler for feature normalization
- **Dimensionality Reduction Options**:
  - PCA with variance threshold (0.99999)
  - Autoencoder option (latent_dim=32)
- **Validation**: Custom CV with competition's wMAE metric

### Key Libraries and Dependencies
- **ML Stack**: 
  - LightGBM (primary model)
  - scikit-learn (preprocessing, utilities)
  - TensorFlow/Keras (neural alternatives, CPU-only)
- **Data Processing**: 
  - pandas, numpy (data manipulation)
  - RDKit (molecular analysis, when available)
- **Visualization**: 
  - matplotlib, seaborn (analysis plots)
- **Utilities**: 
  - tqdm (progress tracking)
  - pathlib (path handling)
  - warnings (suppression)
  - re (SMILES parsing)

## Performance Metrics
- **Current CV Score**: ~0.0541 (competition wMAE)
- **Model Performance Ranking**: LightGBM > Ridge > Transformer
- **Data Efficiency**: Effective handling of 91% missing values
- **Feature Importance**: Target-specific feature selection improves performance

## Running the Code

### Command Line Arguments
```bash
# Standard execution (training + prediction)
python3 model.py

# Cross-validation only (local testing)
python3 model.py --cv-only
python3 model.py --cv

# Kaggle notebook execution
# Auto-detects environment and adjusts paths
```

### Environment Detection
- Automatically detects Kaggle vs local environment
- Adjusts data paths and configurations accordingly
- Handles offline constraints for Kaggle submissions

### Key Requirements
- **Offline Execution**: Must run without internet (Kaggle constraint)
- **CPU-Only**: TensorFlow configured for CPU to ensure reproducibility
- **Deterministic Seeds**: Random seeds set for reproducibility
- **Path Management**: Environment-aware path handling
- **Memory Efficiency**: Optimized for Kaggle's resource limits

## Testing Infrastructure
Comprehensive pytest test suite covering:
- Data loading and integration
- Feature engineering validation
- Model training and prediction
- Cross-validation workflows
- Residual analysis functions
- Environment configuration

Run tests with: `pytest tests/`

## Git Configuration
- **Commit Style**: Single-sentence messages describing the change
- **No Co-authoring**: Clean commits without attribution
- **Atomic Commits**: One logical change per commit
- **Author Info**: 4luqard (turhan.m.tas@gmail.com)
- **Example Format**: "Add polymer crystallinity feature extraction"

## Development Guidelines
1. **Maintain Modularity**: Keep functions focused and reusable
2. **Document Changes**: Update docstrings and comments for complex logic
3. **Kaggle Compatibility**: Always test offline execution capability
4. **Run Tests**: Execute test suite before committing changes
5. **Follow Patterns**: Maintain consistency with existing code style
6. **Preserve Reproducibility**: Use fixed seeds and deterministic operations
7. **Optimize Performance**: Consider memory and compute constraints
8. **Feature Engineering**: Focus on polymer-specific domain knowledge

## Current Development Focus
- Feature engineering refinement for polymer-specific properties
- Advanced missing value imputation strategies
- Model ensemble optimization and stacking
- Cross-validation strategy improvements
- Hyperparameter tuning for each target
- Submission file generation and validation
- Performance profiling and optimization

## Important Notes for Agents
- This is a Kaggle code competition requiring offline execution
- Missing values are due to measurement limitations, not intentional
- Test set requires predictions for ALL 5 targets per sample
- Feature engineering is critical for performance
- Target-specific modeling outperforms multi-output approaches
- Environment detection is automatic - no manual configuration needed
- Always validate submission format before final submission

This project demonstrates production-quality machine learning engineering with emphasis on reproducibility, maintainability, and performance optimization for polymer property prediction from molecular structure.