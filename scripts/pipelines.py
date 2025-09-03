"""
Enhanced ML pipelines for Ames Housing dataset with comprehensive preprocessing.
Integrates statistical data cleaning with advanced feature engineering for optimal model performance.

Key features:
- Smart imputation (neighborhood-based for LotFrontage, statistical for others)
- Statistical outlier handling using IQR method (3.0 multiplier)
- Feature engineering (interactions, ratios, derived features)
- Enhanced skewness correction with outlier handling
- Data type optimization for memory efficiency
- Categorical encoding (OneHot for linear models, Label for tree-based)
- Feature scaling for linear models
- Proper CV-safe preprocessing to prevent data leakage
- Model-specific pipeline configurations

Usage:
1. Run domain_data_cleaning.py on raw data first (one-time domain-specific cleaning)
2. Use make_pipeline() with domain-cleaned data for statistical preprocessing
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler, 
    LabelEncoder,
    PowerTransformer
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import feature builders
try:
    from .feature_builders import BasicFeatureBuilder, AdvancedFeatureBuilder
except ImportError:
    # Handle direct script execution
    from feature_builders import BasicFeatureBuilder, AdvancedFeatureBuilder



class SkewnessCorrector(BaseEstimator, TransformerMixin):
    """
    Advanced skewness correction using multiple transformation methods.
    Based on notebooks/02_feature_engineering/2-3.
    """
    
    def __init__(self, skewness_threshold: float = 0.75, min_improve: float = 0.10):
        self.skewness_threshold = skewness_threshold
        self.min_improve = min_improve
        self.transformations_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            series = X[feature].copy()
            original_skew = abs(series.skew())
            
            if original_skew > self.skewness_threshold:
                # Apply outlier handling using IQR method
                series = self._handle_outliers(series)
                
                # Find best transformation
                method, transformed = self._pick_best_transform(series)
                self.transformations_[feature] = {
                    'method': method,
                    'original_skew': original_skew,
                    'final_skew': abs(transformed.skew()) if method != 'none' else original_skew
                }
                
                # Store PowerTransformer if used
                if method == 'yeojohnson':
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    pt.fit(series.values.reshape(-1, 1))
                    self.transformations_[feature]['transformer'] = pt
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        for feature, transform_info in self.transformations_.items():
            if feature in X.columns:
                series = X[feature].copy()
                method = transform_info['method']
                
                if method != 'none':
                    # Apply outlier handling first
                    series = self._handle_outliers(series)
                    
                    # Apply transformation
                    if method == 'log1p':
                        s_min = series.min()
                        shift = 1 - s_min if s_min <= 0 else 0
                        X[feature] = np.log1p(series + shift)
                    elif method == 'sqrt':
                        s_min = series.min()
                        shift = 1 - s_min if s_min <= 0 else 0
                        X[feature] = np.sqrt(series + shift)
                    elif method == 'cbrt':
                        X[feature] = np.cbrt(series)
                    elif method == 'yeojohnson':
                        transformer = transform_info['transformer']
                        X[feature] = transformer.transform(series.values.reshape(-1, 1)).ravel()
        
        return X
    
    def _handle_outliers(self, s: pd.Series) -> pd.Series:
        """Handle outliers using IQR method with 3.0 multiplier."""
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        return s.clip(lower_bound, upper_bound)
    
    def _transform_candidates(self, s: pd.Series) -> dict:
        out = {'none': s}
        s_min = s.min()
        shift = 1 - s_min if s_min <= 0 else 0
        
        out['log1p'] = np.log1p(s + shift)
        out['sqrt'] = np.sqrt(s + shift)
        out['cbrt'] = np.cbrt(s)
        
        try:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            out['yeojohnson'] = pd.Series(
                pt.fit_transform(s.values.reshape(-1, 1)).ravel(), 
                index=s.index
            )
        except Exception:
            pass
            
        return out
    
    def _pick_best_transform(self, s: pd.Series):
        candidates = self._transform_candidates(s)
        skews = {
            name: abs(val.skew()) 
            for name, val in candidates.items() 
            if np.isfinite(val).all()
        }
        
        if not skews:
            return 'none', s
            
        best_name = min(skews, key=skews.get)
        base = abs(s.skew())
        best = skews[best_name]
        
        # Apply only if improves significantly
        if base > 0 and best <= (1 - self.min_improve) * base:
            return best_name, candidates[best_name]
        else:
            return 'none', s


# ==================== ADVANCED IMPUTATION ====================

class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Intelligent imputation that combines statistical methods with domain knowledge.
    Handles different columns with appropriate strategies.
    """
    
    def __init__(self):
        self.impute_strategies_ = {}
        self.neighborhood_medians_ = {}
        self.column_medians_ = {}
        self.column_modes_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        # LotFrontage: neighborhood-based median
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            self.neighborhood_medians_['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].median().to_dict()
            self.impute_strategies_['LotFrontage'] = 'neighborhood_median'
        
        # Other numeric columns: overall median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'LotFrontage':
                self.column_medians_[col] = X[col].median()
                self.impute_strategies_[col] = 'median'
        
        # Categorical columns: mode with fallback
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            mode_val = X[col].mode()
            if len(mode_val) > 0:
                self.column_modes_[col] = mode_val[0]
            else:
                # Fallback for columns with no mode
                unique_vals = X[col].dropna().unique()
                self.column_modes_[col] = unique_vals[0] if len(unique_vals) > 0 else 'Unknown'
            self.impute_strategies_[col] = 'mode'
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Handle LotFrontage with neighborhood-based median
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            for neighborhood in X['Neighborhood'].unique():
                mask = X['Neighborhood'] == neighborhood
                median_val = self.neighborhood_medians_['LotFrontage'].get(neighborhood, X['LotFrontage'].median())
                X.loc[mask, 'LotFrontage'] = X.loc[mask, 'LotFrontage'].fillna(median_val)
        
        # Handle other numeric columns
        for col, strategy in self.impute_strategies_.items():
            if col in X.columns and strategy == 'median':
                X[col] = X[col].fillna(self.column_medians_[col])
            elif col in X.columns and strategy == 'mode':
                X[col] = X[col].fillna(self.column_modes_[col])
        
        # Final safety check - fill any remaining NaN with simple strategies
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna('Unknown')
                else:
                    X[col] = X[col].fillna(X[col].median())
                
        return X


class StatisticalOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Statistical outlier handling using IQR method.
    Applies winsorization (clipping) rather than removal to preserve data.
    """
    
    def __init__(self, iqr_multiplier: float = 3.0, target_columns: list = None):
        self.iqr_multiplier = iqr_multiplier
        self.target_columns = target_columns or ['GrLivArea', 'LotArea', 'TotalBsmtSF']
        self.bounds_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        for col in self.target_columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                self.bounds_[col] = {'lower': lower_bound, 'upper': upper_bound}
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        for col, bounds in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(bounds['lower'], bounds['upper'])
                
        return X


class DataTypeOptimizer(BaseEstimator, TransformerMixin):
    """
    Optimize data types for memory efficiency and model performance.
    """
    
    def __init__(self):
        self.categorical_numeric = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold']
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Convert numeric columns that represent categories
        for col in self.categorical_numeric:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        return X


# ==================== PREPROCESSORS ====================

def build_preprocessor(
    model_type: str = 'linear',
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build advanced preprocessor with proper categorical encoding.
    
    Args:
        model_type: 'linear' (OneHot) or 'tree' (Label encoding)
        scale_numeric: Whether to scale numeric features
    """
    
    # Numeric pipeline
    num_steps = []
    if scale_numeric:
        num_steps.append(('scale', StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else 'passthrough'
    
    # Categorical pipeline
    if model_type == 'linear':
        # OneHot encoding for linear models (sparse_output=False for feature selection compatibility)
        cat_pipe = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    else:
        # Label encoding for tree-based models
        cat_pipe = Pipeline([
            ('label', LabelEncoder())
        ])
    
    return ColumnTransformer(
        transformers=[
            ('num', num_pipe, selector(dtype_include=np.number)),
            ('cat', cat_pipe, selector(dtype_include=['object', 'category'])),
        ],
        remainder='drop'
    )


# Custom Label Encoder for ColumnTransformer
class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder that handles multiple columns and unseen categories."""
    
    def __init__(self):
        self.encoders_ = {}
        
    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self
        
    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        result = X.copy()
        for col in X.columns:
            if col in self.encoders_:
                le = self.encoders_[col]
                # Handle unseen categories by assigning them to a default value (0)
                col_values = X[col].astype(str)
                encoded_values = []
                for val in col_values:
                    if val in le.classes_:
                        encoded_values.append(le.transform([val])[0])
                    else:
                        # Assign unseen categories to 0 (or could use -1 or len(classes))
                        encoded_values.append(0)
                result[col] = encoded_values
        return result.values


def build_tree_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Preprocessor specifically for tree-based models."""
    
    num_steps = []
    if scale_numeric:
        num_steps.append(('scale', StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else 'passthrough'
    
    cat_pipe = Pipeline([
        ('label', MultiLabelEncoder())
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', num_pipe, selector(dtype_include=np.number)),
            ('cat', cat_pipe, selector(dtype_include=['object', 'category'])),
        ],
        remainder='drop'
    )


# ==================== FEATURE SELECTION ====================

def build_feature_selector(
    selector_type: Optional[str] = None, 
    **kwargs
) -> Optional[BaseEstimator]:
    """
    Build feature selector.
    
    Args:
        selector_type: 'kbest', 'lasso', or None
        **kwargs: Parameters for the selector
    """
    if selector_type is None:
        return None
    elif selector_type == 'kbest':
        k = kwargs.get('k', 'all')
        return SelectKBest(score_func=f_regression, k=k)
    elif selector_type == 'lasso':
        lcv = LassoCV(
            cv=kwargs.get('cv', 5),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1,
            max_iter=2000
        )
        threshold = kwargs.get('threshold', 'median')
        return SelectFromModel(lcv, threshold=threshold)
    else:
        raise ValueError(f"Unknown selector_type: {selector_type}")


# ==================== PIPELINE BUILDERS ====================

def make_pipeline(
    model: BaseEstimator,
    *,
    model_type: str = 'linear',
    feature_eng: Optional[str] = 'basic',
    add_skewness_correction: bool = True,
    handle_outliers: bool = True,
    optimize_dtypes: bool = True,
    scale_numeric: bool = True,
    selector_type: Optional[str] = None,
    selector_kwargs: Optional[Dict[str, Any]] = None,
    target_log: bool = True
) -> BaseEstimator:
    """
    Create enhanced ML pipeline with comprehensive preprocessing.
    
    Args:
        model: Base estimator
        model_type: 'linear' or 'tree' for appropriate preprocessing
        feature_eng: Feature engineering type ('basic', 'advanced', or None)
        add_skewness_correction: Whether to add skewness correction
        handle_outliers: Whether to add statistical outlier handling
        optimize_dtypes: Whether to optimize data types
        scale_numeric: Whether to scale numeric features
        selector_type: Feature selection method
        selector_kwargs: Parameters for feature selector
        target_log: Whether to apply log transformation to target
        
    Returns:
        Complete pipeline (with optional target transformation)
    """
    
    steps = []
    
    # Use SmartImputer but with fallback handling
    steps.append(('smart_impute', SmartImputer()))
    
    # Statistical outlier handling
    if handle_outliers:
        steps.append(('outlier_handler', StatisticalOutlierHandler()))
    
    # Feature engineering
    if feature_eng == 'basic':
        steps.append(('feature_eng', BasicFeatureBuilder()))
    elif feature_eng == 'advanced':
        steps.append(('feature_eng', AdvancedFeatureBuilder()))
    # If feature_eng is None, skip feature engineering step
    
    # Skewness correction (now includes enhanced outlier handling)
    if add_skewness_correction:
        steps.append(('skew_correct', SkewnessCorrector()))
    
    # Data type optimization
    if optimize_dtypes:
        steps.append(('dtype_opt', DataTypeOptimizer()))
    
    # Preprocessing (encoding and scaling only, imputation handled earlier)
    if model_type == 'tree':
        preprocessor = build_tree_preprocessor(scale_numeric=scale_numeric)
    else:
        preprocessor = build_preprocessor(
            model_type=model_type, 
            scale_numeric=scale_numeric
        )
    steps.append(('preprocess', preprocessor))
    
    # Feature selection
    selector = build_feature_selector(selector_type, **(selector_kwargs or {}))
    if selector is not None:
        steps.append(('select', selector))
    
    # Model
    steps.append(('model', model))
    
    # Create pipeline
    pipeline = Pipeline(steps)
    
    # Wrap with target transformation if requested
    if target_log:
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
    
    return pipeline


# ==================== CROSS-VALIDATION UTILITIES ====================

def evaluate_pipeline_cv(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: List[str] = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate pipeline using cross-validation with multiple metrics.
    
    Args:
        pipeline: ML pipeline to evaluate
        X: Features
        y: Target
        cv: Number of CV folds
        scoring: List of scoring metrics
        random_state: Random state for reproducibility
        verbose: Whether to print results
        
    Returns:
        Dictionary with mean and std for each metric
    """
    from sklearn.model_selection import cross_validate
    
    # Setup CV
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Run cross-validation
    cv_results = cross_validate(
        pipeline, X, y, 
        cv=cv_splitter, 
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Process results
    results = {}
    for score_name in scoring:
        test_scores = cv_results[f'test_{score_name}']
        
        # Convert negative scores back to positive for RMSE/MAE
        if score_name.startswith('neg_'):
            test_scores = -test_scores
            metric_name = score_name.replace('neg_', '').upper()
        else:
            metric_name = score_name.upper()
            
        results[metric_name] = {
            'mean': float(np.mean(test_scores)),
            'std': float(np.std(test_scores)),
            'scores': test_scores.tolist()
        }
    
    if verbose:
        print(f"Cross-Validation Results ({cv}-fold):")
        print("-" * 40)
        for metric, values in results.items():
            print(f"{metric:20s}: {values['mean']:.4f} (+/- {values['std']:.4f})")
        print("-" * 40)
    
    return results


def compare_models_cv(
    models_dict: Dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Args:
        models_dict: Dictionary of {model_name: pipeline}
        X: Features
        y: Target
        cv: Number of CV folds
        random_state: Random state
        
    Returns:
        DataFrame with comparison results
    """
    
    results = []
    
    for model_name, pipeline in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        cv_results = evaluate_pipeline_cv(
            pipeline, X, y, cv=cv, random_state=random_state, verbose=False
        )
        
        result_row = {'Model': model_name}
        for metric, values in cv_results.items():
            result_row[f'{metric}_mean'] = values['mean']
            result_row[f'{metric}_std'] = values['std']
        
        results.append(result_row)
    
    comparison_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Sort by RMSE (lower is better)
    if 'ROOT_MEAN_SQUARED_ERROR_mean' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('ROOT_MEAN_SQUARED_ERROR_mean')
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return comparison_df


# ==================== CONVENIENCE FUNCTIONS ====================

def get_default_cv(n_splits: int = 5, random_state: int = 42) -> KFold:
    """Get default cross-validation splitter."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare data from CSV file.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        Tuple of (features, target)
    """
    df = pd.read_csv(data_path)
    
    # Assume SalePrice is the target
    if 'SalePrice' in df.columns:
        target = df['SalePrice']
        features = df.drop('SalePrice', axis=1)
    else:
        raise ValueError("SalePrice column not found in data")
    
    return features, target


# ==================== EXAMPLE CONFIGURATIONS ====================

def get_linear_config() -> Dict[str, Any]:
    """Get configuration for linear regression models."""
    return {
        'model_type': 'linear',
        'feature_eng': 'basic',
        'add_skewness_correction': True,
        'handle_outliers': True,
        'optimize_dtypes': True,
        'scale_numeric': True,
        'selector_type': 'kbest',
        'selector_kwargs': {'k': 20},
        'target_log': True
    }


def get_ridge_config() -> Dict[str, Any]:
    """Get configuration for ridge regression models."""
    return {
        'model_type': 'linear',
        'feature_eng': 'basic',
        'add_skewness_correction': True,
        'handle_outliers': True,
        'optimize_dtypes': True,
        'scale_numeric': True,
        'selector_type': 'lasso',
        'selector_kwargs': {'threshold': 'median'},
        'target_log': True
    }


def get_rf_config() -> Dict[str, Any]:
    """Get configuration for random forest models."""
    return {
        'model_type': 'tree',
        'feature_eng': 'basic',
        'add_skewness_correction': False,  # RF handles skewness well
        'handle_outliers': True,  # Still beneficial for RF
        'optimize_dtypes': True,
        'scale_numeric': False,  # RF doesn't need scaling
        'selector_type': None,  # RF does implicit feature selection
        'selector_kwargs': None,
        'target_log': True
    }


def get_advanced_linear_config() -> Dict[str, Any]:
    """Get configuration for linear regression with advanced features."""
    return {
        'model_type': 'linear',
        'feature_eng': 'advanced',  # Safe advanced features only
        'add_skewness_correction': True,
        'handle_outliers': True,
        'optimize_dtypes': True,
        'scale_numeric': True,
        'selector_type': 'kbest',
        'selector_kwargs': {'k': 25},  # Slightly more features due to advanced FE
        'target_log': True
    }


def get_advanced_ridge_config() -> Dict[str, Any]:
    """Get configuration for ridge regression with advanced features."""
    return {
        'model_type': 'linear',
        'feature_eng': 'advanced',  # Safe advanced features only
        'add_skewness_correction': True,
        'handle_outliers': True,
        'optimize_dtypes': True,
        'scale_numeric': True,
        'selector_type': 'lasso',
        'selector_kwargs': {'threshold': 'median'},
        'target_log': True
    }


def get_advanced_rf_config() -> Dict[str, Any]:
    """Get configuration for random forest with RF-optimized features."""
    return {
        'model_type': 'tree',
        'feature_eng': 'advanced',  # RF-optimized categorical features
        'add_skewness_correction': False,  # RF handles skewness naturally
        'handle_outliers': False,  # RF is robust to outliers, outlier handling can hurt
        'optimize_dtypes': True,
        'scale_numeric': False,  # RF doesn't need scaling
        'selector_type': None,  # RF does implicit feature selection
        'selector_kwargs': None,
        'target_log': True
    }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    print("Advanced ML Pipelines for Ames Housing Dataset")
    print("=" * 50)
    
    # This would be used in practice:
    # X, y = load_and_prepare_data('data/cleaned/domain_cleaned.csv')
    # 
    # from sklearn.linear_model import LinearRegression, Ridge
    # from sklearn.ensemble import RandomForestRegressor
    # 
    # models = {
    #     'Linear': make_pipeline(LinearRegression(), **get_linear_config()),
    #     'Ridge': make_pipeline(Ridge(alpha=1.0), **get_ridge_config()),
    #     'RandomForest': make_pipeline(
    #         RandomForestRegressor(n_estimators=100, random_state=42), 
    #         **get_rf_config()
    #     )
    # }
    # 
    # results = compare_models_cv(models, X, y, cv=5)
    
    print("Enhanced Pipeline configurations available:")
    print("Basic Feature Engineering:")
    print("- get_linear_config(): For Linear Regression")
    print("- get_ridge_config(): For Ridge Regression") 
    print("- get_rf_config(): For Random Forest")
    print("\nAdvanced Feature Engineering:")
    print("- get_advanced_linear_config(): For Linear Regression with advanced features")
    print("- get_advanced_ridge_config(): For Ridge Regression with advanced features")
    print("- get_advanced_rf_config(): For Random Forest with advanced features")
    print("\nUse make_pipeline() to create enhanced pipelines with:")
    print("  • Smart imputation (neighborhood-based for LotFrontage)")
    print("  • Statistical outlier handling (IQR method)")
    print("  • Basic or Advanced feature engineering (feature_eng='basic'/'advanced'/None)")
    print("  • Skewness correction and data type optimization")
    print("Use compare_models_cv() to compare multiple models")
    print("\nFor domain-specific cleaning, run domain_data_cleaning.py first")
