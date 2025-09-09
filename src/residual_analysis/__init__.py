import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import pickle
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')


def should_run_analysis() -> bool:
    """Check if residual analysis should run (only in local environment)"""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is None


def update_gitignore(gitignore_path: str = ".gitignore") -> None:
    """Update gitignore to exclude residual analysis outputs"""
    patterns = [
        "residual_analysis/",
        "*.pkl",
        "residual_plots/",
        "residual_results/",
    ]
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        for pattern in patterns:
            if pattern not in content:
                content += f"\n{pattern}"
        
        with open(gitignore_path, 'w') as f:
            f.write(content.strip() + "\n")


class ResidualAnalysis:
    """Base class for residual analysis of model predictions"""
    
    def __init__(self, output_dir: str = "residual_analysis", save_txt: bool = False, save_png: bool = False):
        self.output_dir = output_dir
        self.targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.save_txt = save_txt
        self.save_png = save_png
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Ensure output directory exists"""
        if should_run_analysis():
            os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_residuals(self, y_true: Dict[str, np.ndarray], 
                          y_pred: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate residuals for each target, handling missing values"""
        residuals = {}
        
        for target in self.targets:
            if target in y_true and target in y_pred:
                mask = ~np.isnan(y_true[target])
                residuals[target] = y_pred[target][mask] - y_true[target][mask]
            
        return residuals
    
    def visualize_residuals(self, residuals: Dict[str, np.ndarray], 
                          fold: Optional[int] = None) -> None:
        """Create residual plots for each target"""
        if not should_run_analysis() or not self.save_png:
            return
            
        plot_dir = os.path.join(self.output_dir, "plots") if self.output_dir != "residual_analysis" else self.output_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        for target, res in residuals.items():
            if len(res) == 0:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Residual Analysis for {target}' + 
                        (f' (Fold {fold})' if fold is not None else ''))
            
            # Histogram
            ax = axes[0, 0]
            ax.hist(res, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            
            # Q-Q plot
            ax = axes[0, 1]
            stats.probplot(res, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot')
            
            # Residuals vs Index
            ax = axes[1, 0]
            ax.scatter(range(len(res)), res, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Sample Index')
            
            # Box plot
            ax = axes[1, 1]
            ax.boxplot(res)
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Box Plot')
            
            plt.tight_layout()
            
            filename = f"residuals_{target}"
            if fold is not None:
                filename += f"_fold_{fold}"
            plt.savefig(os.path.join(self.output_dir, f"{filename}.png"))
            plt.close()
    
    def analyze_patterns(self, residuals: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Analyze statistical patterns in residuals"""
        patterns = {}
        
        for target, res in residuals.items():
            if len(res) == 0:
                continue
                
            patterns[target] = {
                'mean': np.mean(res),
                'std': np.std(res),
                'skewness': stats.skew(res),
                'kurtosis': stats.kurtosis(res),
                'min': np.min(res),
                'max': np.max(res),
                'q25': np.percentile(res, 25),
                'q50': np.percentile(res, 50),
                'q75': np.percentile(res, 75),
                'n_samples': len(res)
            }
            
        return patterns
    
    def analyze_cv_results(self, cv_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze residuals from cross-validation results"""
        results = {}
        
        for fold_name, fold_data in cv_predictions.items():
            y_true = fold_data['y_true']
            y_pred = fold_data['y_pred']
            
            residuals = self.calculate_residuals(y_true, y_pred)
            patterns = self.analyze_patterns(residuals)
            
            results[fold_name] = {
                'residuals': residuals,
                'patterns': patterns
            }
            
            # Visualize if in local environment
            if should_run_analysis():
                fold_idx = int(fold_name.split('_')[-1])
                self.visualize_residuals(residuals, fold=fold_idx)
        
        return results
    
    def save_results(self, results: Any, model_name: str, save_pkl: bool = False, save_json: bool = False, append_mode: bool = True) -> None:
        """Save residual analysis results in multiple formats
        
        Args:
            results: Results to save
            model_name: Model name for file naming
            save_pkl: Whether to save pickle files (default: False)
            save_json: Whether to save JSON files (default: False)
            append_mode: Whether to append to existing files for per-target results (default: True)
        """
        if not should_run_analysis():
            return
        
        # Check if results contain per-target residual data
        # Only use per-target saving when:
        # 1. append_mode is True AND
        # 2. results has 'residuals' key (new format from analyzers)
        if append_mode and isinstance(results, dict) and 'residuals' in results and isinstance(results['residuals'], dict):
            # Save per-target results (append mode)
            # Get residuals from either format
            residuals_data = results['residuals'] if 'residuals' in results else results
            
            for target in self.targets:
                if target in residuals_data:
                    target_results = {
                        'model_name': model_name,
                        'target': target,
                        'residuals': residuals_data[target],
                        'statistics': results.get('statistics', {}).get(target, {}) if 'statistics' in results else {},
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    # Save JSON only if requested
                    if save_json:
                        # Convert to JSON serializable
                        json_target_results = self._convert_to_json_serializable(target_results)
                        
                        # Save JSON (append mode)
                        json_path = os.path.join(self.output_dir, f"residual_analysis_{target}.json")
                        if append_mode and os.path.exists(json_path):
                            # Load existing data
                            with open(json_path, 'r') as f:
                                existing_data = json.load(f)
                            # Ensure it's a list
                            if not isinstance(existing_data, list):
                                existing_data = [existing_data]
                            # Append new data
                            existing_data.append(json_target_results)
                            # Save updated data
                            with open(json_path, 'w') as f:
                                json.dump(existing_data, f, indent=2)
                        else:
                            # Create new file with list
                            with open(json_path, 'w') as f:
                                json.dump([json_target_results], f, indent=2)
                    
                    # Save text (append mode) only if save_txt is True
                    if self.save_txt:
                        txt_path = os.path.join(self.output_dir, f"residual_analysis_{target}.txt")
                        with open(txt_path, 'a' if append_mode else 'w') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"Model: {model_name}\n")
                            f.write(f"Target: {target}\n")
                            f.write(f"Timestamp: {pd.Timestamp.now().isoformat()}\n")
                            f.write(self._format_target_results_as_text(target_results))
                            f.write(f"\n")
                    
                    # Save pkl if requested
                    if save_pkl:
                        pkl_path = os.path.join(self.output_dir, f"residual_analysis_{target}.pkl")
                        if append_mode and os.path.exists(pkl_path):
                            # Load existing data
                            with open(pkl_path, 'rb') as f:
                                existing_data = pickle.load(f)
                            # Ensure it's a list
                            if not isinstance(existing_data, list):
                                existing_data = [existing_data]
                            # Append new data
                            existing_data.append(target_results)
                            # Save updated data
                            with open(pkl_path, 'wb') as f:
                                pickle.dump(existing_data, f)
                        else:
                            # Create new file with list
                            with open(pkl_path, 'wb') as f:
                                pickle.dump([target_results], f)
        else:
            # Save complete results (original behavior for non-per-target results)
            base_name = f"residual_analysis_{model_name}"
            
            # For backward compatibility, save pkl for complete results when not specified
            if save_pkl or (save_pkl is None and not append_mode):
                # Save as pickle (only if requested)
                pkl_path = os.path.join(self.output_dir, f"{base_name}.pkl")
                with open(pkl_path, 'wb') as f:
                    pickle.dump(results, f)
            
            # Save as JSON only if requested
            if save_json:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._convert_to_json_serializable(results)
                
                # Save as JSON
                json_path = os.path.join(self.output_dir, f"{base_name}.json")
                with open(json_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
            
            # Save as human-readable text only if save_txt is True
            if self.save_txt:
                txt_path = os.path.join(self.output_dir, f"{base_name}.txt")
                with open(txt_path, 'w') as f:
                    f.write(self._format_results_as_text(results, model_name))
    
    def save_cv_fold_results(self, fold_data: Dict[str, Any], model_name: str, 
                           fold_idx: int, cv_seed: int) -> None:
        """Save CV fold results in markdown format with combined stats and visualizations"""
        if not should_run_analysis():
            return
        
        # Extract data
        residuals = fold_data.get('residuals', {})
        statistics = fold_data.get('statistics', {})
        
        # Save results for each target
        for target in self.targets:
            if target not in residuals:
                continue
            
            # File path for this target
            md_path = os.path.join(self.output_dir, f"residuals_{target}.md")
            
            # Create markdown content
            content = []
            content.append("---")
            content.append(f"Model: {model_name}")
            content.append(f"Seed: {cv_seed}")
            content.append("")
            content.append("  Statistics:")
            
            # Add statistics if available
            if target in statistics:
                stats = statistics[target]
                for stat_name, stat_val in stats.items():
                    if isinstance(stat_val, (int, float, np.number)):
                        content.append(f"    {stat_name}: {stat_val:.6f}")
                    else:
                        content.append(f"    {stat_name}: {stat_val}")
            
            content.append("")
            content.append("  Visualization:")
            
            # Generate visualization
            if len(residuals[target]) > 0:
                # Create residual plots
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                res = residuals[target]
                
                # Histogram
                ax = axes[0, 0]
                ax.hist(res, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Residuals')
                ax.set_ylabel('Frequency')
                ax.set_title('Residual Distribution')
                
                # Q-Q plot
                ax = axes[0, 1]
                from scipy import stats as scipy_stats
                scipy_stats.probplot(res, dist="norm", plot=ax)
                ax.set_title('Q-Q Plot')
                
                # Residuals vs Index
                ax = axes[1, 0]
                ax.scatter(range(len(res)), res, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Residuals')
                ax.set_title('Residuals vs Sample Index')
                
                # Box plot
                ax = axes[1, 1]
                ax.boxplot(res)
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Box Plot')
                
                plt.tight_layout()
                
                # Save to base64 for embedding
                import io
                import base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode()
                plt.close(fig)
                
                # Add image to markdown
                content.append(f"  ![Residual Analysis](data:image/png;base64,{image_base64})")
            
            content.append("")
            content.append("---")
            content.append("")
            
            # Append to file
            with open(md_path, 'a') as f:
                f.write('\n'.join(content))
    
    def save_cv_results(self, results: Dict[str, Any], model_name: str) -> None:
        """Save CV results in markdown format"""
        if not should_run_analysis():
            return
        
        for target, target_data in results.items():
            if 'statistics' not in target_data:
                continue
            
            # Create simple markdown file with statistics
            md_path = os.path.join(self.output_dir, f"residuals_cv_{model_name}_{target}.md")
            
            content = []
            content.append("---")
            content.append(f"Model: cv_{model_name}")
            content.append("")
            content.append("  Statistics:")
            
            stats = target_data['statistics']
            for stat_name, stat_val in stats.items():
                if isinstance(stat_val, (int, float, np.number)):
                    content.append(f"    {stat_name}: {stat_val:.6f}")
                else:
                    content.append(f"    {stat_name}: {stat_val}")
            
            content.append("")
            content.append("---")
            
            with open(md_path, 'w') as f:
                f.write('\n'.join(content))

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='list')
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if 'float' in str(type(obj)) else int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _format_results_as_text(self, results: Any, model_name: str) -> str:
        """Format results as human-readable text"""
        lines = []
        lines.append(f"Residual Analysis Results - {model_name}")
        lines.append("=" * 50)
        lines.append("")
        
        if isinstance(results, dict):
            for key, value in results.items():
                lines.append(f"\n{key.upper()}:")
                lines.append("-" * 30)
                
                if key == 'residuals' and isinstance(value, dict):
                    for target, residuals in value.items():
                        lines.append(f"\n  {target}:")
                        if isinstance(residuals, np.ndarray):
                            lines.append(f"    Number of samples: {len(residuals)}")
                            lines.append(f"    Mean residual: {np.mean(residuals):.6f}")
                            lines.append(f"    Std residual: {np.std(residuals):.6f}")
                            lines.append(f"    Min residual: {np.min(residuals):.6f}")
                            lines.append(f"    Max residual: {np.max(residuals):.6f}")
                
                elif key == 'patterns' and isinstance(value, dict):
                    for target, patterns in value.items():
                        lines.append(f"\n  {target}:")
                        if isinstance(patterns, dict):
                            for metric, val in patterns.items():
                                lines.append(f"    {metric}: {val:.6f}" if isinstance(val, float) else f"    {metric}: {val}")
                
                elif key == 'statistics' and isinstance(value, dict):
                    lines.append("\n  Statistics:")
                    for target, stats in value.items():
                        lines.append(f"\n    {target}:")
                        if isinstance(stats, dict):
                            for stat_name, stat_val in stats.items():
                                lines.append(f"      {stat_name}: {stat_val:.6f}")
                        else:
                            lines.append(f"      Mean Squared Error: {stats:.6f}")
                
                elif key == 'feature_importance' and isinstance(value, np.ndarray):
                    lines.append(f"  Shape: {value.shape}")
                    lines.append(f"  Top 5 values: {value[:5].tolist() if len(value) > 5 else value.tolist()}")
                    lines.append(f"  Mean importance: {np.mean(value):.6f}")
                
                elif isinstance(value, dict):
                    lines.extend(self._format_dict_as_text(value, indent=2))
                
                elif isinstance(value, np.ndarray):
                    lines.append(f"  Shape: {value.shape}")
                    lines.append(f"  Data type: {value.dtype}")
                    if value.size < 10:
                        lines.append(f"  Values: {value.tolist()}")
                    else:
                        lines.append(f"  First 5 values: {value.flat[:5].tolist()}")
                
                else:
                    lines.append(f"  {value}")
        
        return "\n".join(lines)
    
    def _format_target_results_as_text(self, target_results: Dict[str, Any]) -> str:
        """Format per-target results as human-readable text"""
        lines = []
        
        if 'residuals' in target_results:
            residuals = target_results['residuals']
            if isinstance(residuals, (list, np.ndarray)):
                residuals = np.array(residuals)
                lines.append(f"  Number of samples: {len(residuals)}")
                lines.append(f"  Mean residual: {np.mean(residuals):.6f}")
                lines.append(f"  Std residual: {np.std(residuals):.6f}")
                lines.append(f"  Min residual: {np.min(residuals):.6f}")
                lines.append(f"  Max residual: {np.max(residuals):.6f}")
        
        if 'statistics' in target_results and isinstance(target_results['statistics'], dict):
            lines.append(f"\n  Statistics:")
            for metric, value in target_results['statistics'].items():
                if isinstance(value, (int, float, np.number)):
                    lines.append(f"    {metric}: {value:.6f}")
                else:
                    lines.append(f"    {metric}: {value}")
        
        return "\n".join(lines)
    
    def _format_dict_as_text(self, d: dict, indent: int = 0) -> List[str]:
        """Format nested dictionary as indented text"""
        lines = []
        indent_str = "  " * indent
        
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(self._format_dict_as_text(value, indent + 1))
            elif isinstance(value, (list, np.ndarray)):
                lines.append(f"{indent_str}{key}: {len(value)} items")
            elif isinstance(value, (float, np.float32, np.float64)):
                lines.append(f"{indent_str}{key}: {value:.6f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        
        return lines


class ResidualAnalyzer:
    """Base class for model/method-specific analyzers"""
    
    def __init__(self, model_type: str = None, output_dir: str = "residual_analysis", 
                 save_txt: bool = False, save_png: bool = False):
        self.model_type = model_type
        self.residual_analysis = ResidualAnalysis(output_dir=output_dir)
        self.output_dir = output_dir
        self.save_txt = save_txt
        self.save_png = save_png
    
    def analyze(self, *args, **kwargs):
        """Analyze residuals for specific model/method"""
        raise NotImplementedError
    
    def save_results(self, results: Any, filename_prefix: str, save_pkl: bool = False, save_json: bool = False) -> None:
        """Save results using the residual analysis save method"""
        self.residual_analysis.save_results(results, filename_prefix, save_pkl=save_pkl, save_json=save_json, append_mode=True)
    
    def get_residuals_dataframe(self, X: np.ndarray, y_true: Dict[str, np.ndarray], 
                               y_pred: Dict[str, np.ndarray], smiles: List[str],
                               target: str, method: str = None, is_cv: bool = False, 
                               fold: int = None, seed: int = None) -> pd.DataFrame:
        """Create dataframe with residuals, features, and metadata"""
        # Calculate residuals for the target
        if target in y_true and target in y_pred:
            mask = ~np.isnan(y_true[target])
            residuals = y_pred[target][mask] - y_true[target][mask]
            
            # Create dataframe with features
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X[mask], columns=feature_cols)
            
            # Add SMILES, target, and residuals
            df['SMILES'] = [smiles[i] for i in range(len(mask)) if mask[i]]
            df[target] = y_true[target][mask]
            df['residuals'] = residuals
            
            # Add metadata columns
            if method is not None:
                df['method'] = method
            df['cv'] = is_cv
            if fold is not None:
                df['fold'] = fold
            if seed is not None:
                df['seed'] = seed
                
            return df
        else:
            return pd.DataFrame()
    
    def get_combined_residuals_dataframe(self, X_train: np.ndarray, X_val: np.ndarray,
                                        y_train: Dict[str, np.ndarray], y_val: Dict[str, np.ndarray],
                                        y_pred_train: Dict[str, np.ndarray], y_pred_val: Dict[str, np.ndarray],
                                        train_smiles: List[str], val_smiles: List[str],
                                        target: str, **kwargs) -> pd.DataFrame:
        """Combine train and validation residuals with train_val indicator"""
        # Get train dataframe
        df_train = self.get_residuals_dataframe(
            X_train, y_train, y_pred_train, train_smiles, target, **kwargs
        )
        if not df_train.empty:
            df_train['train_val'] = False
        
        # Get validation dataframe
        df_val = self.get_residuals_dataframe(
            X_val, y_val, y_pred_val, val_smiles, target, **kwargs
        )
        if not df_val.empty:
            df_val['train_val'] = True
        
        # Combine dataframes
        if not df_train.empty and not df_val.empty:
            return pd.concat([df_train, df_val], ignore_index=True)
        elif not df_train.empty:
            return df_train
        elif not df_val.empty:
            return df_val
        else:
            return pd.DataFrame()
    
    def get_preprocessing_residuals_dataframe(self, original_features: np.ndarray,
                                             transformed_features: np.ndarray,
                                             smiles: List[str], targets: Dict[str, np.ndarray],
                                             residuals: np.ndarray, method: str) -> pd.DataFrame:
        """Create dataframe for preprocessing methods with original and transformed features"""
        # Create dataframe with original features
        orig_cols = [f'orig_feature_{i}' for i in range(original_features.shape[1])]
        df = pd.DataFrame(original_features, columns=orig_cols)
        
        # Add transformed features
        trans_cols = [f'trans_feature_{i}' for i in range(transformed_features.shape[1])]
        df_trans = pd.DataFrame(transformed_features, columns=trans_cols)
        df = pd.concat([df, df_trans], axis=1)
        
        # Add SMILES and targets
        df['SMILES'] = smiles
        for target_name, target_values in targets.items():
            df[target_name] = target_values
        
        # Add residuals and metadata
        df['residuals'] = residuals
        df['method'] = method
        
        return df
    
    def save_residuals_dataframe(self, df: pd.DataFrame, method: str, target: str, 
                                fold: int = None) -> str:
        """Save residuals dataframe as parquet file"""
        if not should_run_analysis():
            return ""
        
        # Create filename
        if fold is not None:
            filename = f"residuals_{method}_{target}_fold_{fold}.parquet"
        else:
            filename = f"residuals_{method}_{target}.parquet"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save as parquet (skip if pyarrow not available)
        try:
            df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        except ImportError:
            # If pyarrow not available, save as CSV instead
            csv_filename = filename.replace('.parquet', '.csv')
            filepath = os.path.join(self.output_dir, csv_filename)
            df.to_csv(filepath, index=False)
        
        return filepath


class LightGBMResidualAnalyzer(ResidualAnalyzer):
    """Residual analyzer for LightGBM models"""
    
    def analyze(self, predictions: Optional[Any] = None, actuals: Optional[Any] = None, 
                model: Optional[Any] = None, X: Optional[Any] = None, y: Optional[Any] = None, 
                model_name: str = None) -> Dict[str, Any]:
        """Analyze LightGBM model residuals and feature importance"""
        results = {}
        
        # Use actuals parameter if provided, otherwise fallback to y
        if actuals is not None:
            y = actuals
            
        if predictions is not None and y is not None:
            # Convert predictions and y to dict format if needed
            if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
                y_pred = {target: predictions[:, i] 
                         for i, target in enumerate(self.residual_analysis.targets)}
            else:
                y_pred = predictions
                
            if isinstance(y, pd.DataFrame):
                y_true = {target: y[target].values 
                         for target in self.residual_analysis.targets}
            else:
                y_true = y
            
            residuals = self.residual_analysis.calculate_residuals(y_true, y_pred)
            results['residuals'] = residuals
            results['patterns'] = self.residual_analysis.analyze_patterns(residuals)
        else:
            # Return empty residuals if no y provided
            results['residuals'] = {}
        
        # Add feature importance if available
        if hasattr(model, 'feature_importance'):
            results['feature_importance'] = model.feature_importance()
        
        return results


class TransformerResidualAnalyzer(ResidualAnalyzer):
    """Residual analyzer for Transformer models"""
    
    def analyze(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Transformer predictions including attention patterns"""
        results = {}
        
        if 'predictions' in predictions:
            # Analyze prediction residuals
            # Note: y_true should be passed separately in real implementation
            results['residuals'] = predictions.get('residuals', {})
        
        if 'attention_weights' in predictions:
            # Analyze attention patterns
            attention = predictions['attention_weights']
            results['attention_analysis'] = {
                'mean_attention': np.mean(attention, axis=0),
                'attention_entropy': -np.sum(attention * np.log(attention + 1e-9), axis=-1).mean()
            }
        
        if 'hidden_states' in predictions:
            # Analyze hidden states
            hidden = predictions['hidden_states']
            results['hidden_analysis'] = {
                'mean_activation': np.mean(hidden),
                'std_activation': np.std(hidden)
            }
        
        return results


class AutoencoderResidualAnalyzer(ResidualAnalyzer):
    """Residual analyzer for Autoencoder models"""
    
    def analyze(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Autoencoder reconstruction and latent space"""
        results = {}
        
        if 'reconstructed' in outputs and 'original' in outputs:
            # Calculate reconstruction error
            reconstruction_error = np.mean((outputs['reconstructed'] - outputs['original'])**2, axis=1)
            results['reconstruction_error'] = {
                'mean': np.mean(reconstruction_error),
                'std': np.std(reconstruction_error),
                'max': np.max(reconstruction_error),
                'percentile_95': np.percentile(reconstruction_error, 95)
            }
        elif 'reconstructed' in outputs:
            # If no original provided, still return empty reconstruction_error
            results['reconstruction_error'] = {}
        
        if 'latent' in outputs:
            # Analyze latent representations
            latent = outputs['latent']
            results['latent_analysis'] = {
                'mean_activation': np.mean(latent, axis=0),
                'std_activation': np.std(latent, axis=0),
                'sparsity': np.mean(np.abs(latent) < 0.1)
            }
        
        return results


class PCAResidualAnalyzer(ResidualAnalyzer):
    """Residual analyzer for PCA transformations"""
    
    def analyze(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PCA reconstruction and variance"""
        results = {}
        
        if 'transformed' in outputs and 'original' in outputs:
            # Calculate reconstruction error if inverse transform available
            if 'reconstructed' in outputs:
                reconstruction_error = np.mean((outputs['reconstructed'] - outputs['original'])**2, axis=1)
                results['reconstruction_error'] = {
                    'mean': np.mean(reconstruction_error),
                    'std': np.std(reconstruction_error)
                }
        else:
            # Return empty reconstruction_error if original not provided
            results['reconstruction_error'] = {}
        
        if 'explained_variance' in outputs:
            # Analyze variance explanation
            var = outputs['explained_variance']
            results['variance_analysis'] = {
                'cumulative_variance': np.cumsum(var),
                'n_components_95': np.argmax(np.cumsum(var) >= 0.95) + 1
            }
        
        return results


class PLSResidualAnalyzer(ResidualAnalyzer):
    """Residual analyzer for PLS transformations"""
    
    def analyze(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PLS components and predictions"""
        results = {}
        
        if 'components' in outputs:
            # Analyze component loadings
            components = outputs['components']
            results['component_analysis'] = {
                'mean_loading': np.mean(np.abs(components), axis=1),
                'max_loading': np.max(np.abs(components), axis=1)
            }
        
        if 'scores' in outputs:
            # Analyze scores
            scores = outputs['scores']
            results['score_analysis'] = {
                'variance': np.var(scores, axis=0),
                'correlation': np.corrcoef(scores.T) if scores.shape[1] > 1 else None
            }
        
        return results


# Import integration helpers
from .integration import (
    ResidualAnalysisHook,
    with_residual_analysis,
    integrate_with_cv,
    integrate_with_model,
    integrate_with_data_processing
)

# Export main classes
__all__ = [
    'ResidualAnalysis',
    'ResidualAnalyzer',
    'LightGBMResidualAnalyzer', 
    'TransformerResidualAnalyzer',
    'AutoencoderResidualAnalyzer',
    'PCAResidualAnalyzer',
    'PLSResidualAnalyzer',
    'should_run_analysis',
    'update_gitignore',
    'ResidualAnalysisHook',
    'with_residual_analysis',
    'integrate_with_cv',
    'integrate_with_model',
    'integrate_with_data_processing'
]