"""
CV Diagnostic Tracking System

This module provides detailed diagnostic tracking for cross-validation to help
understand and debug CV behavior, especially the discrepancy between CV and 
public leaderboard scores.
"""

import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CVDiagnostics:
    """Cross-validation diagnostic tracker for detailed analysis."""
    
    def __init__(self, output_dir: str = "output/cv_diagnostics"):
        """Initialize CV diagnostics tracker."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diagnostics = {
            "session_id": self.session_id,
            "folds": [],
            "summary": {},
            "data_stats": {},
            "feature_stats": {},
            "predictions": []
        }
    
    def track_data_split(self, fold: int, train_idx: np.ndarray, val_idx: np.ndarray,
                         train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Track data split statistics for each fold."""
        fold_stats = {
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_missing_pattern": {},
            "val_missing_pattern": {}
        }
        
        # Track missing value patterns
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        for target in targets:
            if target in train_data.columns:
                train_missing = train_data[target].isna().sum()
                val_missing = val_data[target].isna().sum()
                
                fold_stats["train_missing_pattern"][target] = {
                    "missing_count": int(train_missing),
                    "missing_pct": float(train_missing / len(train_data) * 100)
                }
                fold_stats["val_missing_pattern"][target] = {
                    "missing_count": int(val_missing),
                    "missing_pct": float(val_missing / len(val_data) * 100)
                }
        
        self.diagnostics["folds"].append(fold_stats)
    
    def track_target_training(self, fold: int, target: str, 
                             train_samples: int, val_samples: int,
                             features_used: List[str], 
                             alpha_value: float):
        """Track training details for each target in each fold."""
        target_info = {
            "target": target,
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "features_used": features_used,
            "alpha": float(alpha_value),
            "feature_count": len(features_used)
        }
        
        # Find the fold and add target info
        for fold_data in self.diagnostics["folds"]:
            if fold_data["fold"] == fold:
                if "targets" not in fold_data:
                    fold_data["targets"] = []
                fold_data["targets"].append(target_info)
                break
    
    def track_predictions(self, fold: int, target: str, 
                         val_indices: np.ndarray,
                         predictions: np.ndarray, 
                         actuals: np.ndarray):
        """Track predictions for detailed analysis."""
        pred_data = {
            "fold": fold,
            "target": target,
            "predictions": predictions.tolist(),
            "actuals": actuals.tolist(),
            "indices": val_indices.tolist(),
            "errors": (predictions - actuals).tolist(),
            "mae": float(np.mean(np.abs(predictions - actuals))),
            "rmse": float(np.sqrt(np.mean((predictions - actuals)**2)))
        }
        
        self.diagnostics["predictions"].append(pred_data)
    
    def track_fold_score(self, fold: int, score: float, 
                        target_scores: Dict[str, float]):
        """Track scores for each fold."""
        for fold_data in self.diagnostics["folds"]:
            if fold_data["fold"] == fold:
                fold_data["score"] = float(score)
                fold_data["target_scores"] = {k: float(v) for k, v in target_scores.items()}
                break
    
    def track_feature_statistics(self, features_df: pd.DataFrame):
        """Track feature statistics across the entire dataset."""
        self.diagnostics["feature_stats"] = {
            "total_features": len(features_df.columns),
            "feature_names": list(features_df.columns),
            "missing_by_feature": {}
        }
        
        for col in features_df.columns:
            missing = features_df[col].isna().sum()
            self.diagnostics["feature_stats"]["missing_by_feature"][col] = {
                "missing_count": int(missing),
                "missing_pct": float(missing / len(features_df) * 100)
            }
    
    def finalize_session(self, mean_score: float, std_score: float, 
                        individual_scores: List[float]):
        """Finalize the diagnostic session with summary statistics."""
        self.diagnostics["summary"] = {
            "mean_cv_score": float(mean_score),
            "std_cv_score": float(std_score),
            "individual_fold_scores": [float(s) for s in individual_scores],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save diagnostics to file
        output_file = self.output_dir / f"cv_diagnostics_{self.session_id}.json"
        with open(output_file, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)
        
        print(f"\nCV Diagnostics saved to: {output_file}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        return output_file
    
    def _generate_visualizations(self):
        """Generate diagnostic visualizations."""
        try:
            # Create figure directory
            fig_dir = self.output_dir / f"figures_{self.session_id}"
            fig_dir.mkdir(exist_ok=True)
            
            # 1. Fold scores distribution
            fold_scores = [f["score"] for f in self.diagnostics["folds"] if "score" in f]
            if fold_scores:
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(fold_scores)), fold_scores)
                plt.axhline(y=np.mean(fold_scores), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(fold_scores):.4f}')
                plt.xlabel('Fold')
                plt.ylabel('Score (lower is better)')
                plt.title('CV Scores by Fold')
                plt.legend()
                plt.tight_layout()
                plt.savefig(fig_dir / 'fold_scores.png')
                plt.close()
            
            # 2. Missing data patterns
            targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            train_missing = {t: [] for t in targets}
            val_missing = {t: [] for t in targets}
            
            for fold in self.diagnostics["folds"]:
                for target in targets:
                    if target in fold.get("train_missing_pattern", {}):
                        train_missing[target].append(
                            fold["train_missing_pattern"][target]["missing_pct"]
                        )
                        val_missing[target].append(
                            fold["val_missing_pattern"][target]["missing_pct"]
                        )
            
            if any(train_missing.values()):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Train missing patterns
                x = np.arange(len(targets))
                width = 0.15
                for i, fold_data in enumerate(zip(*[train_missing[t] for t in targets])):
                    if fold_data:
                        ax1.bar(x + i*width, fold_data, width, label=f'Fold {i}')
                
                ax1.set_xlabel('Target')
                ax1.set_ylabel('Missing %')
                ax1.set_title('Training Data Missing Patterns')
                ax1.set_xticks(x + width * 2)
                ax1.set_xticklabels(targets)
                ax1.legend()
                
                # Val missing patterns
                for i, fold_data in enumerate(zip(*[val_missing[t] for t in targets])):
                    if fold_data:
                        ax2.bar(x + i*width, fold_data, width, label=f'Fold {i}')
                
                ax2.set_xlabel('Target')
                ax2.set_ylabel('Missing %')
                ax2.set_title('Validation Data Missing Patterns')
                ax2.set_xticks(x + width * 2)
                ax2.set_xticklabels(targets)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(fig_dir / 'missing_patterns.png')
                plt.close()
            
            # 3. Sample sizes by target and fold
            fig, ax = plt.subplots(figsize=(12, 8))
            
            fold_target_samples = []
            labels = []
            
            for fold_data in self.diagnostics["folds"]:
                if "targets" in fold_data:
                    for target_data in fold_data["targets"]:
                        fold_target_samples.append({
                            'Fold': fold_data["fold"],
                            'Target': target_data["target"],
                            'Train': target_data["train_samples"],
                            'Val': target_data["val_samples"]
                        })
            
            if fold_target_samples:
                df_samples = pd.DataFrame(fold_target_samples)
                df_pivot = df_samples.pivot_table(
                    index='Target', 
                    columns='Fold', 
                    values=['Train', 'Val'], 
                    aggfunc='first'
                )
                
                df_pivot.plot(kind='bar', ax=ax)
                ax.set_ylabel('Sample Count')
                ax.set_title('Sample Sizes by Target and Fold')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(fig_dir / 'sample_sizes.png')
                plt.close()
            
            print(f"Diagnostic visualizations saved to: {fig_dir}")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report of the diagnostics."""
        report = []
        report.append("=" * 60)
        report.append(f"CV DIAGNOSTIC SUMMARY - Session: {self.session_id}")
        report.append("=" * 60)
        
        if "summary" in self.diagnostics:
            report.append(f"\nOverall CV Score: {self.diagnostics['summary']['mean_cv_score']:.4f} "
                         f"(+/- {self.diagnostics['summary']['std_cv_score']:.4f})")
            report.append(f"Individual Fold Scores: {self.diagnostics['summary']['individual_fold_scores']}")
        
        report.append("\n" + "-" * 40)
        report.append("FOLD-WISE ANALYSIS")
        report.append("-" * 40)
        
        for fold_data in self.diagnostics["folds"]:
            report.append(f"\nFold {fold_data['fold']}:")
            report.append(f"  Score: {fold_data.get('score', 'N/A')}")
            report.append(f"  Train Size: {fold_data['train_size']}")
            report.append(f"  Val Size: {fold_data['val_size']}")
            
            if "targets" in fold_data:
                report.append("  Target Training Details:")
                for target in fold_data["targets"]:
                    report.append(f"    {target['target']}: "
                                f"{target['train_samples']} train, "
                                f"{target['val_samples']} val, "
                                f"{target['feature_count']} features, "
                                f"alpha={target['alpha']}")
        
        report.append("\n" + "-" * 40)
        report.append("POTENTIAL ISSUES DETECTED")
        report.append("-" * 40)
        
        # Check for data imbalance
        fold_scores = [f.get("score", 0) for f in self.diagnostics["folds"] if "score" in f]
        if fold_scores:
            score_std = np.std(fold_scores)
            if score_std > 0.01:  # Threshold for high variance
                report.append(f"⚠️  High variance in fold scores (std={score_std:.4f})")
        
        # Check for sample size variations
        train_sizes = [f["train_size"] for f in self.diagnostics["folds"]]
        if len(set(train_sizes)) > 1:
            report.append(f"⚠️  Inconsistent train sizes across folds: {train_sizes}")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / f"cv_report_{self.session_id}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nSummary report saved to: {report_file}")
        
        return report_text