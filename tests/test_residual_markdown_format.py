import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import base64
import re
from unittest.mock import Mock, patch, MagicMock
from src.residual_analysis import ResidualAnalysis


class TestResidualMarkdownFormat:
    """Test residual analysis markdown output format according to user specifications."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1.1, 2.1, 3.1, 4.1, 5.1]))
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name='Tg')
        return X, y
    
    @pytest.fixture
    def analyzer(self):
        """Create ResidualAnalysis instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ResidualAnalysis(output_dir=tmpdir)
            yield analyzer
    
    def test_markdown_file_naming_convention(self, analyzer, sample_data, mock_model):
        """Test that residual files follow the naming convention: residuals_{target}.md"""
        X, y = sample_data
        
        # Mock the residual analysis to return statistics
        mock_stats = {
            'mean': 0.1,
            'std': 0.05,
            'mse': 0.01,
            'rmse': 0.1,
            'mae': 0.08,
            'max_error': 0.2,
            'min_error': -0.15
        }
        
        with patch('src.residual_analysis.should_run_analysis', return_value=True):
            # Create fold data
            fold_data = {
                'residuals': {'Tg': np.array([0.1, -0.05, 0.2, -0.1, 0.15])},
                'statistics': {'Tg': mock_stats}
            }
            
            # Call the method
            analyzer.save_cv_fold_results(fold_data, 'lightgbm', 0, 42)
            
            # Check that the file follows correct naming convention
            residual_file = Path(analyzer.output_dir) / "residuals_Tg.md"
            wrong_file = Path(analyzer.output_dir) / "residuals_cv_lightgbm_Tg.md"
            
            # The current implementation uses wrong naming, so this should fail
            assert residual_file.exists(), "Residual file should be named residuals_Tg.md"
            assert not wrong_file.exists(), "File should not use residuals_cv_lightgbm_Tg.md naming"
    
    def test_markdown_format_structure(self, analyzer, sample_data, mock_model):
        """Test that markdown format follows the specified structure."""
        X, y = sample_data
        
        mock_stats = {
            'mean': 0.1,
            'std': 0.05,
            'mse': 0.01,
            'rmse': 0.1,
            'mae': 0.08,
            'max_error': 0.2,
            'min_error': -0.15
        }
        
        # Mock matplotlib to avoid creating actual plots
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('src.residual_analysis.should_run_analysis', return_value=True):
            
            # Create fold data
            fold_data = {
                'residuals': {'Tg': np.array([0.1, -0.05, 0.2, -0.1, 0.15])},
                'statistics': {'Tg': mock_stats}
            }
            
            # Call the method
            analyzer.save_cv_fold_results(fold_data, 'lightgbm', 0, 42)
            
            # Read the file (it uses correct naming now)
            residual_file = Path(analyzer.output_dir) / "residuals_Tg.md"
            assert residual_file.exists(), "File should exist"
            
            content = residual_file.read_text()
            
            # Check format structure
            assert "---" in content, "Missing YAML frontmatter"
            assert "Model: lightgbm" in content, "Model should be 'lightgbm'"
            assert "Seed: 42" in content, "Should use 'Seed:' with correct value"
            
            # Check sections
            assert "Statistics:" in content
            assert "Visualization:" in content
            
            # Validate correct format structure
            expected_pattern = r"---\s*\nModel: lightgbm\s*\nSeed: 42\s*\n\s*Statistics:\s*\n.*\n\s*Visualization:\s*\n"
            assert re.search(expected_pattern, content, re.DOTALL), "Markdown format does not match expected structure"
    
    def test_statistics_are_included(self, analyzer, sample_data, mock_model):
        """Test that residual analysis statistics are properly included."""
        X, y = sample_data
        
        mock_stats = {
            'mean': 0.1,
            'std': 0.05,
            'mse': 0.01,
            'rmse': 0.1,
            'mae': 0.08,
            'max_error': 0.2,
            'min_error': -0.15
        }
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('src.residual_analysis.should_run_analysis', return_value=True):
            
            # Create fold data  
            fold_data = {
                'residuals': {'Tg': np.array([0.1, -0.05, 0.2, -0.1, 0.15])},
                'statistics': {'Tg': mock_stats}
            }
            
            # Call the method
            analyzer.save_cv_fold_results(fold_data, 'lightgbm', 0, 42)
            
            # Read the file
            residual_file = Path(analyzer.output_dir) / "residuals_Tg.md"
            assert residual_file.exists(), "File should exist"
            
            content = residual_file.read_text()
            
            # Check statistics are included
            assert "Statistics:" in content
            assert "mean:" in content
            assert "std:" in content
            assert "rmse:" in content 
            assert "mae:" in content
    
    def test_visualization_embedded_as_base64(self, analyzer, sample_data, mock_model):
        """Test that visualizations are embedded as base64 in markdown."""
        X, y = sample_data
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('src.residual_analysis.should_run_analysis', return_value=True):
            
            # Create fold data
            fold_data = {
                'residuals': {'Tg': np.array([0.1, -0.05, 0.2, -0.1, 0.15])},
                'statistics': {'Tg': {'mean': 0.1}}
            }
            
            # Call the method
            analyzer.save_cv_fold_results(fold_data, 'lightgbm', 0, 42)
            
            # Read the file
            residual_file = Path(analyzer.output_dir) / "residuals_Tg.md"
            assert residual_file.exists(), "File should exist"
            
            content = residual_file.read_text()
            
            # Check for visualization section and base64 image
            assert "Visualization:" in content
            assert "![Residual Analysis]" in content
            assert "data:image/png;base64," in content
    
    def test_no_json_files_by_default(self, analyzer, sample_data, mock_model):
        """Test that JSON files are not created by default."""
        X, y = sample_data
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            with patch('src.residual_analysis.should_run_analysis', return_value=True):
                fold_data = {
                    'residuals': {'Tg': np.array([0.1, -0.05, 0.2])},
                    'statistics': {'Tg': {'mean': 0.1}}
                }
                analyzer.save_cv_fold_results(fold_data, 'lightgbm', 0, 42)
            
            # Check that no JSON files were created
            json_files = list(Path(analyzer.output_dir).glob("*.json"))
            assert len(json_files) == 0, "JSON files should not be created by default"
    
    def test_append_mode_for_multiple_folds(self, analyzer, sample_data, mock_model):
        """Test that multiple folds append to the same markdown file."""
        X, y = sample_data
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            with patch('src.residual_analysis.should_run_analysis', return_value=True):
                # Save results for multiple folds
                for fold in range(3):
                    for seed in [42, 123]:
                        fold_data = {
                            'residuals': {'Tg': np.array([0.1 + fold*0.01, -0.05, 0.2])},
                            'statistics': {'Tg': {'mean': 0.1 + fold*0.01}}
                        }
                        analyzer.save_cv_fold_results(fold_data, 'lightgbm', fold, seed)
            
            residual_file = Path(analyzer.output_dir) / "residuals_Tg.md"
            if residual_file.exists():
                content = residual_file.read_text()
                
                # Should have 6 entries (3 folds × 2 seeds)
                assert content.count("Model: lightgbm") == 6, "Should have 6 fold entries"
                assert content.count("Seed: 42") == 3, "Should have 3 entries for seed 42"
                assert content.count("Seed: 123") == 3, "Should have 3 entries for seed 123"
    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])