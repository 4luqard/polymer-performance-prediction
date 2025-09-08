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
    
    def __init__(self, enable: bool = True):
        self.enable = enable and should_run_analysis()
        self.base_analyzer = ResidualAnalysis() if self.enable else None
        self.analyzers = {}
        
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