"""
Utility to run cross-validation multiple times for more robust results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import json
from datetime import datetime
import os


def run_cv_multiple_times(cv_function, X, y, n_runs: int = 5, cv_folds: int = 5, 
                          target_columns: List[str] = None, seed_offset: int = 0):
    """
    Run cross-validation multiple times with different random seeds.
    
    Args:
        cv_function: The cross-validation function to call
        X: Features
        y: Targets
        n_runs: Number of times to run CV
        cv_folds: Number of folds for each CV run
        target_columns: List of target columns
        seed_offset: Offset for random seeds (useful for continuing runs)
    
    Returns:
        Dictionary with aggregated results
    """
    print(f"\n{'='*60}")
    print(f"Running {n_runs} independent CV runs with {cv_folds} folds each")
    print(f"Total validation evaluations: {n_runs * cv_folds}")
    print(f"{'='*60}\n")
    
    all_results = []
    all_scores = []
    
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"CV RUN {run + 1}/{n_runs}")
        print(f"{'='*50}")
        
        # Set different random seed for each run
        np.random.seed(42 + run + seed_offset)
        
        # Run CV
        cv_result = cv_function(
            X, y, 
            cv_folds=cv_folds, 
            target_columns=target_columns,
            enable_diagnostics=(run == 0)  # Only enable diagnostics on first run
        )
        
        if cv_result:
            all_results.append(cv_result)
            all_scores.extend(cv_result['fold_scores'])
            
            print(f"\nRun {run + 1} Summary:")
            print(f"  Mean: {cv_result['cv_mean']:.4f}")
            print(f"  Std:  {cv_result['cv_std']:.4f}")
            print(f"  Range: [{min(cv_result['fold_scores']):.4f}, {max(cv_result['fold_scores']):.4f}]")
    
    # Aggregate results
    if all_results:
        run_means = [r['cv_mean'] for r in all_results]
        run_stds = [r['cv_std'] for r in all_results]
        
        aggregated_results = {
            'n_runs': n_runs,
            'cv_folds': cv_folds,
            'total_evaluations': n_runs * cv_folds,
            'overall_mean': np.mean(all_scores),
            'overall_std': np.std(all_scores),
            'overall_min': np.min(all_scores),
            'overall_max': np.max(all_scores),
            'run_means': run_means,
            'run_stds': run_stds,
            'mean_of_means': np.mean(run_means),
            'std_of_means': np.std(run_means),
            'all_fold_scores': all_scores,
            'individual_runs': all_results
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Number of CV runs: {n_runs}")
        print(f"Folds per run: {cv_folds}")
        print(f"Total evaluations: {n_runs * cv_folds}")
        print(f"\nAggregated Results:")
        print(f"  Overall mean score: {aggregated_results['overall_mean']:.4f}")
        print(f"  Overall std dev:    {aggregated_results['overall_std']:.4f}")
        print(f"  Score range:        [{aggregated_results['overall_min']:.4f}, {aggregated_results['overall_max']:.4f}]")
        print(f"\nRun-level Statistics:")
        print(f"  Mean of run means:  {aggregated_results['mean_of_means']:.4f}")
        print(f"  Std of run means:   {aggregated_results['std_of_means']:.4f}")
        
        # Check for high variance
        if aggregated_results['std_of_means'] > 0.005:
            print(f"\n⚠️  WARNING: High variance between runs detected!")
            print(f"    This suggests the CV results may be unstable.")
        
        # Save results
        output_dir = "output/cv_multiple_runs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/cv_runs_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Create visualization
        _plot_cv_results(aggregated_results, output_dir, timestamp)
        
        return aggregated_results
    
    return None


def _plot_cv_results(results: Dict, output_dir: str, timestamp: str):
    """Create visualization of CV results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Distribution of all fold scores
        ax1 = axes[0, 0]
        ax1.hist(results['all_fold_scores'], bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(x=results['overall_mean'], color='red', linestyle='--', 
                   label=f'Mean: {results["overall_mean"]:.4f}')
        ax1.set_xlabel('CV Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of All Fold Scores')
        ax1.legend()
        
        # 2. Run means with error bars
        ax2 = axes[0, 1]
        run_indices = range(1, results['n_runs'] + 1)
        ax2.errorbar(run_indices, results['run_means'], yerr=results['run_stds'], 
                    fmt='o-', capsize=5, capthick=2)
        ax2.axhline(y=results['mean_of_means'], color='red', linestyle='--',
                   label=f'Mean: {results["mean_of_means"]:.4f}')
        ax2.set_xlabel('CV Run')
        ax2.set_ylabel('Mean Score')
        ax2.set_title('CV Scores by Run (with std dev)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot of scores by run
        ax3 = axes[1, 0]
        run_scores = []
        for i, run in enumerate(results['individual_runs']):
            for score in run['fold_scores']:
                run_scores.append({'Run': i+1, 'Score': score})
        
        df_scores = pd.DataFrame(run_scores)
        sns.boxplot(data=df_scores, x='Run', y='Score', ax=ax3)
        ax3.set_xlabel('CV Run')
        ax3.set_ylabel('Score')
        ax3.set_title('Score Distribution by Run')
        
        # 4. Cumulative mean plot
        ax4 = axes[1, 1]
        cumulative_means = np.cumsum(results['all_fold_scores']) / np.arange(1, len(results['all_fold_scores']) + 1)
        ax4.plot(cumulative_means)
        ax4.axhline(y=results['overall_mean'], color='red', linestyle='--',
                   label=f'Final mean: {results["overall_mean"]:.4f}')
        ax4.set_xlabel('Number of Evaluations')
        ax4.set_ylabel('Cumulative Mean Score')
        ax4.set_title('Convergence of Mean Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cv_runs_visualization_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_dir}/cv_runs_visualization_{timestamp}.png")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")