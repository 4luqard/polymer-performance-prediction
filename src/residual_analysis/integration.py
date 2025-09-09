"""Integration hooks for residual analysis with existing codebase"""

from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from .import (
    ResidualAnalysis,
    LightGBMResidualAnalyzer,
    TransformerResidualAnalyzer,
    AutoencoderResidualAnalyzer,
    PCAResidualAnalyzer,
    PLSResidualAnalyzer,
    should_run_analysis
)


class ResidualAnalysisHook:
    """Main integration hook for residual analysis"""
    
    def __init__(self, enable: bool = True, save_txt: bool = False, save_png: bool = False):
        self.enable = enable and should_run_analysis()
        self.save_txt = save_txt
        self.save_png = save_png
        self.base_analyzer = ResidualAnalysis() if self.enable else None
        self.analyzers = {}
        self.dataframes = []  # Store generated dataframes
        
    def register_analyzer(self, name: str, analyzer: Any):
        """Register a model/method-specific analyzer"""
        if self.enable:
            self.analyzers[name] = analyzer
    
    def save(self, results: Any, filename_prefix: str, save_pkl: bool = False, save_json: bool = False) -> None:
        """Save results using the base analyzer"""
        if self.enable and self.base_analyzer:
            self.base_analyzer.save_results(results, filename_prefix, save_pkl=save_pkl, save_json=save_json)
    
    def analyze_predictions(self, 
                          y_true: Any, 
                          y_pred: Any,
                          model_name: str = "model",
                          fold: Optional[int] = None,
                          **kwargs) -> Optional[Dict[str, Any]]:
        """Analyze predictions with residual analysis"""
        if not self.enable:
            return None
            
        results = {}
        
        # Convert to proper format
        if isinstance(y_true, pd.DataFrame):
            y_true_dict = {col: y_true[col].values for col in y_true.columns}
        else:
            y_true_dict = y_true
            
        if isinstance(y_pred, np.ndarray):
            # Assume columns match target order
            targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            y_pred_dict = {targets[i]: y_pred[:, i] for i in range(y_pred.shape[1])}
        else:
            y_pred_dict = y_pred
        
        # Calculate residuals
        residuals = self.base_analyzer.calculate_residuals(y_true_dict, y_pred_dict)
        patterns = self.base_analyzer.analyze_patterns(residuals)
        
        # Visualize if fold is provided
        if fold is not None:
            self.base_analyzer.visualize_residuals(residuals, fold=fold)
        
        results['residuals'] = residuals
        results['patterns'] = patterns
        
        # Save results
        self.base_analyzer.save_results(results, f"{model_name}_fold{fold}" if fold else model_name)
        
        return results
    
    def analyze_model_specific(self, 
                             model_type: str,
                             model: Any = None,
                             outputs: Any = None,
                             **kwargs) -> Optional[Dict[str, Any]]:
        """Analyze model-specific outputs"""
        if not self.enable:
            return None
            
        if model_type not in self.analyzers:
            return None
            
        analyzer = self.analyzers[model_type]
        
        if model_type == 'lightgbm':
            return analyzer.analyze(model, **kwargs)
        elif model_type in ['transformer', 'autoencoder', 'pca', 'pls']:
            return analyzer.analyze(outputs, **kwargs)
        
        return None
    
    def generate_residuals_dataframe(self,
                                    X_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_train: Any,
                                    y_val: Any,
                                    y_pred_train: Any,
                                    y_pred_val: Any,
                                    train_smiles: list,
                                    val_smiles: list,
                                    method: str,
                                    is_cv: bool = False,
                                    fold: int = None,
                                    seed: int = None) -> pd.DataFrame:
        """Generate residuals dataframe combining train and validation data"""
        if not self.enable:
            return pd.DataFrame()
        
        # Create a generic analyzer to use its dataframe methods
        from . import ResidualAnalyzer
        analyzer = ResidualAnalyzer(save_txt=self.save_txt, save_png=self.save_png)
        
        # Convert y_train and y_val to dict format if needed
        if isinstance(y_train, pd.DataFrame):
            y_train_dict = {col: y_train[col].values for col in y_train.columns}
        else:
            y_train_dict = y_train
            
        if isinstance(y_val, pd.DataFrame):
            y_val_dict = {col: y_val[col].values for col in y_val.columns}
        else:
            y_val_dict = y_val
        
        # Convert predictions to dict format if needed
        if isinstance(y_pred_train, pd.DataFrame):
            y_pred_train_dict = {col: y_pred_train[col].values for col in y_pred_train.columns}
        elif isinstance(y_pred_train, np.ndarray) and y_pred_train.ndim == 2:
            targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            y_pred_train_dict = {targets[i]: y_pred_train[:, i] for i in range(y_pred_train.shape[1])}
        else:
            y_pred_train_dict = y_pred_train
            
        if isinstance(y_pred_val, pd.DataFrame):
            y_pred_val_dict = {col: y_pred_val[col].values for col in y_pred_val.columns}
        elif isinstance(y_pred_val, np.ndarray) and y_pred_val.ndim == 2:
            targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            y_pred_val_dict = {targets[i]: y_pred_val[:, i] for i in range(y_pred_val.shape[1])}
        else:
            y_pred_val_dict = y_pred_val
        
        # Generate dataframes for each target
        all_dfs = []
        for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
            df = analyzer.get_combined_residuals_dataframe(
                X_train=X_train,
                X_val=X_val,
                y_train=y_train_dict,
                y_val=y_val_dict,
                y_pred_train=y_pred_train_dict,
                y_pred_val=y_pred_val_dict,
                train_smiles=train_smiles,
                val_smiles=val_smiles,
                target=target,
                method=method,
                is_cv=is_cv,
                fold=fold,
                seed=seed
            )
            
            if not df.empty:
                # Save the dataframe as parquet
                filepath = analyzer.save_residuals_dataframe(df, method, target, fold)
                all_dfs.append(df)
                
        # Store for later access if needed
        self.dataframes.extend(all_dfs)
        
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# Decorator for easy integration
def with_residual_analysis(model_type: str = 'generic'):
    """Decorator to add residual analysis to any prediction function"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Run the original function
            result = func(*args, **kwargs)
            
            # Run residual analysis if enabled
            if should_run_analysis():
                hook = ResidualAnalysisHook()
                
                # Try to extract predictions and ground truth
                if isinstance(result, dict):
                    y_true = result.get('y_true')
                    y_pred = result.get('y_pred') or result.get('predictions')
                    
                    if y_true is not None and y_pred is not None:
                        hook.analyze_predictions(y_true, y_pred, model_name=model_type)
            
            return result
        return wrapper
    return decorator


# CV Integration Helper
def integrate_with_cv(cv_function: Callable) -> Callable:
    """Integrate residual analysis with cross-validation"""
    def wrapper(X, y, *args, **kwargs):
        # Check if residual analysis should run
        enable_residual = kwargs.pop('enable_residual_analysis', should_run_analysis())
        
        if enable_residual:
            hook = ResidualAnalysisHook()
            
            # Wrap the CV function to capture predictions
            original_results = cv_function(X, y, *args, **kwargs)
            
            # Analyze CV results if they contain predictions
            if isinstance(original_results, dict) and 'predictions' in original_results:
                for fold_idx, (y_true, y_pred) in enumerate(original_results['predictions']):
                    hook.analyze_predictions(y_true, y_pred, 
                                           model_name='cv_lightgbm', 
                                           fold=fold_idx)
            
            return original_results
        else:
            return cv_function(X, y, *args, **kwargs)
    
    return wrapper


# Model Integration Helper  
def integrate_with_model(train_function: Callable) -> Callable:
    """Integrate residual analysis with model training"""
    def wrapper(*args, **kwargs):
        # Check if residual analysis should run
        enable_residual = kwargs.pop('enable_residual_analysis', should_run_analysis())
        
        if enable_residual:
            hook = ResidualAnalysisHook()
            hook.register_analyzer('lightgbm', LightGBMResidualAnalyzer())
            
            # Run training
            result = train_function(*args, **kwargs)
            
            # Analyze if we have model and predictions
            if isinstance(result, tuple) and len(result) >= 2:
                model, predictions = result[0], result[1]
                
                # Get validation data if available
                y_val = kwargs.get('y_val')
                if y_val is not None:
                    hook.analyze_predictions(y_val, predictions, model_name='lightgbm_final')
                
                # Analyze model-specific features
                hook.analyze_model_specific('lightgbm', model=model, 
                                          predictions=predictions, y=y_val)
            
            return result
        else:
            return train_function(*args, **kwargs)
    
    return wrapper


# Data Processing Integration Helper
def integrate_with_data_processing(process_function: Callable) -> Callable:
    """Integrate residual analysis with data processing methods"""
    def wrapper(*args, **kwargs):
        # Check if residual analysis should run
        enable_residual = kwargs.pop('enable_residual_analysis', should_run_analysis())
        
        if enable_residual:
            hook = ResidualAnalysisHook()
            hook.register_analyzer('transformer', TransformerResidualAnalyzer())
            hook.register_analyzer('autoencoder', AutoencoderResidualAnalyzer()) 
            hook.register_analyzer('pca', PCAResidualAnalyzer())
            hook.register_analyzer('pls', PLSResidualAnalyzer())
            
            # Run processing
            result = process_function(*args, **kwargs)
            
            # Analyze based on the method used
            method_type = kwargs.get('method_type')
            if method_type and method_type in hook.analyzers:
                hook.analyze_model_specific(method_type, outputs=result)
            
            return result
        else:
            return process_function(*args, **kwargs)
    
    return wrapper