"""
Residual Analysis Module for NeurIPS Polymer Prediction

This module provides comprehensive residual analysis capabilities for all model types.
Designed to work locally like cross-validation, not needed for Kaggle submissions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats


@dataclass
class ResidualMetrics:
    """Container for residual analysis metrics."""
    target: str
    mae: float
    rmse: float
    r2: float
    mean_residual: float
    std_residual: float
    median_residual: float
    q25_residual: float
    q75_residual: float
    iqr_residual: float
    skewness: float
    kurtosis: float
    outliers_count: int
    outliers_percentage: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'target': self.target,
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'mean_residual': self.mean_residual,
            'std_residual': self.std_residual,
            'median_residual': self.median_residual,
            'q25_residual': self.q25_residual,
            'q75_residual': self.q75_residual,
            'iqr_residual': self.iqr_residual,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'outliers_count': self.outliers_count,
            'outliers_percentage': self.outliers_percentage
        }


class ResidualAnalyzer:
    """Comprehensive residual analysis for all model types."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize residual analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir or Path('residual_analysis')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for residuals from different models/methods
        self.residuals_store = {}
        self.predictions_store = {}
        self.true_values_store = {}
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, target_name: str = "unknown") -> ResidualMetrics:
        """
        Compute residual metrics for predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_name: Name of the target variable
            
        Returns:
            ResidualMetrics object with computed metrics
        """
        from scipy import stats
        
        residuals = y_true - y_pred
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        median_residual = np.median(residuals)
        q25_residual = np.percentile(residuals, 25)
        q75_residual = np.percentile(residuals, 75)
        iqr_residual = q75_residual - q25_residual
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        # Detect outliers (residuals beyond 3 standard deviations)
        outliers = np.abs(residuals - mean_residual) > 3 * std_residual
        outliers_count = np.sum(outliers)
        outliers_percentage = outliers_count / len(residuals) * 100
        
        return ResidualMetrics(
            target=target_name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mean_residual=mean_residual,
            std_residual=std_residual,
            median_residual=median_residual,
            q25_residual=q25_residual,
            q75_residual=q75_residual,
            iqr_residual=iqr_residual,
            skewness=skewness,
            kurtosis=kurtosis,
            outliers_count=outliers_count,
            outliers_percentage=outliers_percentage
        )
        
    def compute_residuals(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         model_name: str,
                         method_name: Optional[str] = None) -> np.ndarray:
        """
        Compute residuals for predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model (e.g., 'lightgbm', 'transformer')
            method_name: Name of method (e.g., 'pca', 'autoencoder')
            
        Returns:
            Residuals array
        """
        residuals = y_true - y_pred
        
        # Store for later analysis
        key = f"{model_name}_{method_name}" if method_name else model_name
        self.residuals_store[key] = residuals
        self.predictions_store[key] = y_pred
        self.true_values_store[key] = y_true
        
        return residuals
    
    def analyze_residuals(self, 
                         residuals: np.ndarray,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         target_name: str) -> ResidualMetrics:
        """
        Perform comprehensive residual analysis.
        
        Args:
            residuals: Residuals array
            y_true: True values
            y_pred: Predicted values
            target_name: Name of the target variable
            
        Returns:
            ResidualMetrics object with analysis results
        """
        # Remove NaN values
        mask = ~np.isnan(residuals) & ~np.isnan(y_true) & ~np.isnan(y_pred)
        residuals_clean = residuals[mask]
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(residuals_clean) == 0:
            return None
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Residual statistics
        mean_res = np.mean(residuals_clean)
        std_res = np.std(residuals_clean)
        median_res = np.median(residuals_clean)
        q25_res = np.percentile(residuals_clean, 25)
        q75_res = np.percentile(residuals_clean, 75)
        iqr_res = q75_res - q25_res
        
        # Distribution shape
        skewness = stats.skew(residuals_clean)
        kurtosis_val = stats.kurtosis(residuals_clean)
        
        # Outlier detection (using IQR method)
        lower_bound = q25_res - 1.5 * iqr_res
        upper_bound = q75_res + 1.5 * iqr_res
        outliers = (residuals_clean < lower_bound) | (residuals_clean > upper_bound)
        outliers_count = np.sum(outliers)
        outliers_percentage = 100 * outliers_count / len(residuals_clean)
        
        return ResidualMetrics(
            target=target_name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mean_residual=mean_res,
            std_residual=std_res,
            median_residual=median_res,
            q25_residual=q25_res,
            q75_residual=q75_res,
            iqr_residual=iqr_res,
            skewness=skewness,
            kurtosis=kurtosis_val,
            outliers_count=outliers_count,
            outliers_percentage=outliers_percentage
        )
    
    def analyze_multi_target(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           target_names: List[str],
                           model_name: str,
                           method_name: Optional[str] = None) -> Dict[str, ResidualMetrics]:
        """
        Analyze residuals for multiple targets.
        
        Args:
            y_true: True values (n_samples, n_targets)
            y_pred: Predicted values (n_samples, n_targets)
            target_names: Names of target variables
            model_name: Name of the model
            method_name: Name of method
            
        Returns:
            Dictionary of ResidualMetrics for each target
        """
        results = {}
        
        # Ensure 2D arrays
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Compute residuals
        residuals = self.compute_residuals(y_true, y_pred, model_name, method_name)
        
        # Analyze each target
        for i, target_name in enumerate(target_names):
            target_residuals = residuals[:, i] if residuals.ndim > 1 else residuals
            target_true = y_true[:, i] if y_true.ndim > 1 else y_true
            target_pred = y_pred[:, i] if y_pred.ndim > 1 else y_pred
            
            metrics = self.analyze_residuals(
                target_residuals, 
                target_true, 
                target_pred,
                target_name
            )
            
            if metrics:
                results[target_name] = metrics
        
        return results
    
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      residuals: np.ndarray,
                      target_name: str,
                      model_name: str,
                      save: bool = True) -> None:
        """
        Create comprehensive residual plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            residuals: Residuals
            target_name: Name of target
            model_name: Name of model
            save: Whether to save plots
        """
        # Remove NaN values
        mask = ~np.isnan(residuals) & ~np.isnan(y_true) & ~np.isnan(y_pred)
        residuals_clean = residuals[mask]
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(residuals_clean) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Residual Analysis - {model_name} - {target_name}', fontsize=16)
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred_clean, residuals_clean, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # 2. Q-Q plot
        stats.probplot(residuals_clean, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # 3. Histogram of residuals
        axes[0, 2].hist(residuals_clean, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residual Distribution')
        
        # 4. Predicted vs Actual
        axes[1, 0].scatter(y_true_clean, y_pred_clean, alpha=0.5)
        axes[1, 0].plot([y_true_clean.min(), y_true_clean.max()], 
                       [y_true_clean.min(), y_true_clean.max()], 
                       'r--', lw=2)
        axes[1, 0].set_xlabel('True Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Predicted vs Actual')
        
        # 5. Residuals vs True Values
        axes[1, 1].scatter(y_true_clean, residuals_clean, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('True Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs True Values')
        
        # 6. Box plot of residuals
        axes[1, 2].boxplot(residuals_clean)
        axes[1, 2].set_ylabel('Residuals')
        axes[1, 2].set_title('Residual Box Plot')
        
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / f'residual_plots_{model_name}_{target_name}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved residual plots to {plot_path}")
        
        plt.show()
    
    def save_metrics(self, 
                    metrics: Union[ResidualMetrics, Dict[str, ResidualMetrics]],
                    model_name: str,
                    method_name: Optional[str] = None) -> None:
        """
        Save residual metrics to file.
        
        Args:
            metrics: ResidualMetrics or dict of ResidualMetrics
            model_name: Name of model
            method_name: Name of method
        """
        key = f"{model_name}_{method_name}" if method_name else model_name
        file_path = self.output_dir / f'residual_metrics_{key}.json'
        
        if isinstance(metrics, ResidualMetrics):
            data = metrics.to_dict()
        else:
            data = {k: v.to_dict() for k, v in metrics.items()}
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved residual metrics to {file_path}")
    
    def compare_models_dict(self, models_residuals: Dict[str, np.ndarray]) -> str:
        """
        Compare residuals from different models (simplified version).
        
        Args:
            models_residuals: Dictionary mapping model names to residual arrays
            
        Returns:
            String summary of comparison
        """
        comparison_lines = []
        comparison_lines.append("\nModel Comparison Summary:")
        comparison_lines.append("-" * 50)
        
        metrics_by_model = {}
        for model_name, residuals in models_residuals.items():
            # Compute basic metrics
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            
            metrics_by_model[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'Mean': mean_res,
                'Std': std_res
            }
            
            comparison_lines.append(f"\n{model_name}:")
            comparison_lines.append(f"  MAE: {mae:.4f}")
            comparison_lines.append(f"  RMSE: {rmse:.4f}")
            comparison_lines.append(f"  Mean Residual: {mean_res:.4f}")
            comparison_lines.append(f"  Std Residual: {std_res:.4f}")
        
        # Find best model for each metric
        if len(metrics_by_model) > 1:
            comparison_lines.append("\n" + "="*50)
            comparison_lines.append("Best Models by Metric:")
            comparison_lines.append("-" * 50)
            
            for metric in ['MAE', 'RMSE']:
                best_model = min(metrics_by_model.keys(), 
                               key=lambda x: metrics_by_model[x][metric])
                best_value = metrics_by_model[best_model][metric]
                comparison_lines.append(f"{metric}: {best_model} ({best_value:.4f})")
        
        return "\n".join(comparison_lines)
    
    def compare_models(self, target_names: List[str]) -> pd.DataFrame:
        """
        Compare residuals across different models/methods.
        
        Args:
            target_names: List of target names
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for key in self.residuals_store:
            residuals = self.residuals_store[key]
            y_true = self.true_values_store[key]
            y_pred = self.predictions_store[key]
            
            # Analyze each target
            for i, target in enumerate(target_names):
                if residuals.ndim > 1:
                    target_res = residuals[:, i]
                    target_true = y_true[:, i]
                    target_pred = y_pred[:, i]
                else:
                    target_res = residuals
                    target_true = y_true
                    target_pred = y_pred
                
                # Skip if all NaN
                if np.all(np.isnan(target_res)):
                    continue
                
                metrics = self.analyze_residuals(
                    target_res, 
                    target_true, 
                    target_pred,
                    target
                )
                
                if metrics:
                    comparison_data.append({
                        'model_method': key,
                        'target': target,
                        'mae': metrics.mae,
                        'rmse': metrics.rmse,
                        'r2': metrics.r2,
                        'mean_residual': metrics.mean_residual,
                        'std_residual': metrics.std_residual,
                        'outliers_pct': metrics.outliers_percentage
                    })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_path = self.output_dir / 'model_comparison.csv'
        df.to_csv(comparison_path, index=False)
        print(f"Saved model comparison to {comparison_path}")
        
        return df
    
    def generate_report(self, target_names: List[str]) -> str:
        """
        Generate comprehensive residual analysis report.
        
        Args:
            target_names: List of target names
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("RESIDUAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        for key in self.residuals_store:
            report.append(f"\nModel/Method: {key}")
            report.append("-" * 40)
            
            residuals = self.residuals_store[key]
            y_true = self.true_values_store[key]
            y_pred = self.predictions_store[key]
            
            for i, target in enumerate(target_names):
                if residuals.ndim > 1:
                    target_res = residuals[:, i]
                    target_true = y_true[:, i]
                    target_pred = y_pred[:, i]
                else:
                    target_res = residuals
                    target_true = y_true
                    target_pred = y_pred
                
                # Skip if all NaN
                if np.all(np.isnan(target_res)):
                    continue
                
                metrics = self.analyze_residuals(
                    target_res, 
                    target_true, 
                    target_pred,
                    target
                )
                
                if metrics:
                    report.append(f"\n  Target: {target}")
                    report.append(f"    MAE: {metrics.mae:.4f}")
                    report.append(f"    RMSE: {metrics.rmse:.4f}")
                    report.append(f"    R²: {metrics.r2:.4f}")
                    report.append(f"    Mean Residual: {metrics.mean_residual:.4f}")
                    report.append(f"    Std Residual: {metrics.std_residual:.4f}")
                    report.append(f"    Outliers: {metrics.outliers_count} ({metrics.outliers_percentage:.1f}%)")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / 'residual_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Saved report to {report_path}")
        
        return report_text


def integrate_with_model(model_function):
    """
    Decorator to integrate residual analysis with any model.
    
    Usage:
        @integrate_with_model
        def train_model(X, y, ...):
            # Your model training code
            return predictions
    """
    def wrapper(*args, **kwargs):
        # Extract return_residuals flag if provided
        return_residuals = kwargs.pop('return_residuals', False)
        
        # Run the original model
        predictions = model_function(*args, **kwargs)
        
        if return_residuals:
            # Extract y_true from args (assuming it's the second argument)
            y_true = args[1] if len(args) > 1 else kwargs.get('y', None)
            
            if y_true is not None:
                residuals = y_true - predictions
                return predictions, residuals
        
        return predictions
    
    return wrapper