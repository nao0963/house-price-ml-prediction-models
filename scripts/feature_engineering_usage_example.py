"""
Feature Engineering Usage Example
Demonstrates how to use BasicFeatureBuilder and AdvancedFeatureBuilder with pipelines.

This example shows A/B testing between basic and advanced feature engineering approaches.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import pipeline functions and configurations
from pipelines import (
    make_pipeline, 
    compare_models_cv,
    get_linear_config, get_ridge_config, get_rf_config,
    get_advanced_linear_config, get_advanced_ridge_config, get_advanced_rf_config
)

def example_usage():
    """Example of how to use the feature engineering pipeline."""
    
    # Load your data (replace with actual data loading)
    # X, y = load_and_prepare_data('data/split/train.csv')
    
    print("=== Feature Engineering Pipeline Usage Example ===")
    print()
    
    # Example 1: Basic Feature Engineering Models
    print("1. Basic Feature Engineering Models:")
    basic_models = {
        'Linear_Basic': make_pipeline(LinearRegression(), **get_linear_config()),
        'Ridge_Basic': make_pipeline(Ridge(alpha=1.0), **get_ridge_config()),
        'RF_Basic': make_pipeline(RandomForestRegressor(n_estimators=100, random_state=42), **get_rf_config())
    }
    
    # Example 2: Advanced Feature Engineering Models
    print("2. Advanced Feature Engineering Models:")
    advanced_models = {
        'Linear_Advanced': make_pipeline(LinearRegression(), **get_advanced_linear_config()),
        'Ridge_Advanced': make_pipeline(Ridge(alpha=1.0), **get_advanced_ridge_config()),
        'RF_Advanced': make_pipeline(RandomForestRegressor(n_estimators=100, random_state=42), **get_advanced_rf_config())
    }
    
    # Example 3: Custom Configuration
    print("3. Custom Configuration:")
    custom_pipeline = make_pipeline(
        Ridge(alpha=0.5),
        model_type='linear',
        feature_eng='advanced',  # Use advanced feature engineering
        add_skewness_correction=True,
        handle_outliers=True,
        scale_numeric=True,
        selector_type='lasso',
        selector_kwargs={'threshold': '0.01*mean'},
        target_log=True
    )
    
    # Example 4: No Feature Engineering (baseline)
    print("4. No Feature Engineering (Baseline):")
    baseline_pipeline = make_pipeline(
        LinearRegression(),
        feature_eng=None,  # No feature engineering
        model_type='linear',
        target_log=True
    )
    
    print("Pipeline configurations created successfully!")
    print()
    print("To run comparison:")
    print("# results = compare_models_cv(basic_models, X, y, cv=5)")
    print("# advanced_results = compare_models_cv(advanced_models, X, y, cv=5)")
    print()
    print("Available feature_eng options:")
    print("- 'basic': Standard feature engineering from 03_basic_models")
    print("- 'advanced': Enhanced features from 04_model_improvement")
    print("- None: No feature engineering (baseline)")


def a_b_testing_example():
    """Example of A/B testing basic vs advanced feature engineering."""
    
    print("=== A/B Testing: Basic vs Advanced Feature Engineering ===")
    print()
    
    # For 03_basic_models (use basic feature engineering)
    basic_models_for_03 = {
        'Linear_03': make_pipeline(LinearRegression(), feature_eng='basic', target_log=True),
        'Ridge_03': make_pipeline(Ridge(), feature_eng='basic', target_log=True),
        'RF_03': make_pipeline(RandomForestRegressor(random_state=42), feature_eng='basic', scale_numeric=False, target_log=True)
    }
    
    # For 04_enhanced (use advanced feature engineering)
    advanced_models_for_04 = {
        'Linear_04': make_pipeline(LinearRegression(), feature_eng='advanced', target_log=True),
        'Ridge_04': make_pipeline(Ridge(), feature_eng='advanced', target_log=True),
        'RF_04': make_pipeline(RandomForestRegressor(random_state=42), feature_eng='advanced', scale_numeric=False, target_log=True)
    }
    
    print("Basic models for notebooks/03_basic_models:")
    for name, pipeline in basic_models_for_03.items():
        steps = [step[0] for step in pipeline.regressor.steps]
        print(f"  {name}: {steps}")
    
    print()
    print("Advanced models for notebooks/04_enhanced:")
    for name, pipeline in advanced_models_for_04.items():
        steps = [step[0] for step in pipeline.regressor.steps]
        print(f"  {name}: {steps}")
    
    print()
    print("Usage in notebooks:")
    print("# In 03_basic_models notebooks:")
    print("# pipeline = make_pipeline(model, feature_eng='basic', **other_configs)")
    print()
    print("# In 04_enhanced notebooks:")
    print("# pipeline = make_pipeline(model, feature_eng='advanced', **other_configs)")


if __name__ == "__main__":
    example_usage()
    print("\n" + "="*60 + "\n")
    a_b_testing_example()
