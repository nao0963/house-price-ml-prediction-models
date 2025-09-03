# Enhanced Pipeline Flow Documentation

## Overview
This document explains how the enhanced `make_pipeline()` function works for different model types in the Ames Housing prediction project. The pipeline now integrates statistical data cleaning with configurable feature engineering (Basic vs Advanced) for optimal model performance.

## Two-Stage Data Processing Architecture

### Stage 1: Domain-Specific Cleaning (One-time)
**Script:** `domain_data_cleaning.py`
- Handles domain knowledge-based missing value interpretation
- Removes extreme outliers that are data quality issues
- Basic data type optimization
- **Input:** Split train data → **Output:** Domain-cleaned data

### Stage 2: Statistical Pipeline Processing (CV-safe)
**Script:** `pipelines.py` with `make_pipeline()`
- Smart statistical imputation
- Statistical outlier handling (winsorization)
- Feature engineering and transformation
- Model-specific preprocessing
- **Input:** Domain-cleaned data → **Output:** Model-ready data

## Function Relationships

### Core Functions
- `make_pipeline()`: Enhanced pipeline builder function
- `build_preprocessor()`: General-purpose preprocessor (encoding and scaling only)
- `build_tree_preprocessor()`: Tree-model specific preprocessor
- `get_linear_config()`, `get_ridge_config()`, `get_rf_config()`: Model-specific configurations

### New Processing Classes
- `SmartImputer`: Intelligent imputation with neighborhood-based strategies
- `StatisticalOutlierHandler`: IQR-based outlier handling (winsorization)
- `DataTypeOptimizer`: Memory-efficient data type optimization
- `SkewnessCorrector`: Enhanced skewness correction with outlier handling
- `BasicFeatureBuilder`: Minimal feature engineering (TotalSF, HouseAge)
- `AdvancedFeatureBuilder`: Comprehensive feature engineering (interactions, ratios, derived, categorical)

### Feature Builder Comparison

| Aspect | BasicFeatureBuilder | AdvancedFeatureBuilder |
|--------|-------------------|----------------------|
| **Features Added** | 2 features | 24 features |
| **Derived Features** | TotalSF, HouseAge | TotalBathrooms, EffectiveAge, TotalPorchSF |
| **Interaction Features** | None | Quality×Area, Quality×TotalSF, Bath×Area, Garage×Quality |
| **Ratio Features** | None | Area ratios, AvgRoomSize |
| **Categorical Features** | None | Age categories, boolean indicators, quality/neighborhood categories |
| **Complexity** | Minimal baseline | Comprehensive feature set |
| **Use Case** | Performance baseline | Model optimization |

### Preprocessor Comparison

| Aspect | `build_preprocessor` | `build_tree_preprocessor` |
|--------|---------------------|---------------------------|
| **Model Support** | Linear, Tree models | Tree models only |
| **Categorical Encoding** | Conditional (OneHot vs Label) | Label encoding only |
| **Default Scaling** | `True` (for linear models) | `False` (tree model optimized) |
| **Flexibility** | High (model_type parameter) | Low (tree model fixed) |
| **Label Encoder** | Standard LabelEncoder | Custom MultiLabelEncoder (handles unseen categories) |

#### Categorical Encoding Details

**`build_preprocessor`**:
```python
if model_type == 'linear':
    # OneHot encoding for linear models
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
else:
    # Label encoding for tree-based models  
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('label', LabelEncoder())  # Standard LabelEncoder
    ])
```

**`build_tree_preprocessor`**:
```python
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('label', MultiLabelEncoder())  # Custom MultiLabelEncoder - safer for unseen categories
])
```

## Feature Builder Implementation Details

### BasicFeatureBuilder
**Purpose**: Minimal baseline with only essential features
**Features Created**:
- `TotalSF`: Sum of TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- `HouseAge`: Current year - YearBuilt (clipped to non-negative)

**Implementation**:
```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    
    # Total square footage
    if {'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'}.issubset(X.columns):
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        
    # House age
    if {'YrSold', 'YearBuilt'}.issubset(X.columns):
        current_year = X['YrSold'].max()
        X['HouseAge'] = (current_year - X['YearBuilt']).clip(lower=0)
    
    return X
```

### AdvancedFeatureBuilder
**Purpose**: Comprehensive feature engineering for optimal model performance
**Inherits**: BasicFeatureBuilder (includes TotalSF, HouseAge)

**Feature Categories**:

1. **Interaction Features (4)**:
   - `Quality_x_Area`: OverallQual × GrLivArea
   - `Quality_x_TotalSF`: OverallQual × TotalSF
   - `Bath_x_Area`: TotalBathrooms × GrLivArea
   - `Garage_x_Quality`: GarageCars × OverallQual

2. **Ratio Features (5)**:
   - `GrLivArea_to_LotArea`: Living area efficiency
   - `1stFlr_to_GrLivArea`: First floor proportion
   - `GarageArea_to_GrLivArea`: Garage size relative to living area
   - `BsmtArea_to_GrLivArea`: Basement proportion
   - `AvgRoomSize`: GrLivArea / TotRmsAbvGrd

3. **Derived Features (3)**:
   - `TotalBathrooms`: Full + 0.5×Half + BsmtFull + 0.5×BsmtHalf
   - `EffectiveAge`: YrSold - max(YearBuilt, YearRemodAdd)
   - `TotalPorchSF`: Sum of all porch areas

4. **Categorical Features (12)**:
   - `AgeCategory`: Binned house age (New, Mid, Old, Very_Old)
   - `IsNew`: Built in sale year
   - `Has2ndFloor`, `HasBasement`, `HasFireplace`, `HasPool`, `HasMasVnrArea`, `HasPorch`: Boolean indicators
   - `QualityCategory`: Binned overall quality (Low, Medium, High)
   - `NeighborhoodCategory`: Target-based neighborhood grouping (Low, Medium, High)

**Key Implementation Features**:
- Safe division with epsilon to avoid division by zero
- Infinite value replacement for ratio features
- Target-based neighborhood learning during fit phase
- Category-based data type optimization

## Model-Specific Pipeline Flows

### 1. Linear Regression Model

**Call:**
```python
pipeline = make_pipeline(LinearRegression(), **get_linear_config())
```

**Configuration (`get_linear_config`)**:
```python
{
    'model_type': 'linear',
    'add_feature_engineering': True,
    'add_skewness_correction': True,
    'handle_outliers': True,
    'optimize_dtypes': True,
    'scale_numeric': True,
    'selector_type': 'kbest',
    'selector_kwargs': {'k': 20},
    'target_log': True
}
```

**Function Call Flow:**
```
make_pipeline()
├── 1. SmartImputer() added (intelligent imputation)
│   └── LotFrontage: neighborhood median, others: statistical median/mode
├── 2. StatisticalOutlierHandler() added (handle_outliers=True)
│   └── IQR method with 3.0 multiplier (winsorization)
├── 3. BasicFeatureBuilder() or AdvancedFeatureBuilder() added (add_feature_engineering=True)
│   ├── BasicFeatureBuilder: TotalSF, HouseAge (2 features)
│   └── AdvancedFeatureBuilder: interactions, ratios, derived, categorical (24 features)
├── 4. SkewnessCorrector() added (add_skewness_correction=True)
│   └── Enhanced with IQR-based outlier handling
├── 5. DataTypeOptimizer() added (optimize_dtypes=True)
│   └── Convert categorical numerics to category type
├── 6. build_preprocessor() called (model_type='linear')
│   └── OneHotEncoder + StandardScaler (no imputation)
├── 7. build_feature_selector() called (selector_type='kbest')
│   └── SelectKBest(k=20)
├── 8. LinearRegression() added
└── 9. TransformedTargetRegressor() wrapping (target_log=True)
    └── log1p/expm1 transformation
```

**Final Pipeline Structure:**
```
TransformedTargetRegressor(
  regressor=Pipeline([
    ('smart_impute', SmartImputer),
    ('outlier_handler', StatisticalOutlierHandler),
    ('feature_eng', BasicFeatureBuilder | AdvancedFeatureBuilder),
    ('skew_correct', SkewnessCorrector),
    ('dtype_opt', DataTypeOptimizer),
    ('preprocess', ColumnTransformer[OneHot+StandardScaler]),
    ('select', SelectKBest),
    ('model', LinearRegression)
  ]),
  func=log1p, inverse_func=expm1
)
```

### 2. Ridge Regression Model

**Call:**
```python
pipeline = make_pipeline(Ridge(alpha=1.0), **get_ridge_config())
```

**Configuration (`get_ridge_config`)**:
```python
{
    'model_type': 'linear',
    'add_feature_engineering': True,
    'add_skewness_correction': True,
    'handle_outliers': True,
    'optimize_dtypes': True,
    'scale_numeric': True,
    'selector_type': 'lasso',          # ← Different from Linear!
    'selector_kwargs': {'threshold': 'median'},  # ← Different from Linear!
    'target_log': True
}
```

**Function Call Flow:**
```
make_pipeline()
├── 1. SmartImputer() added (intelligent imputation)
├── 2. StatisticalOutlierHandler() added (handle_outliers=True)
├── 3. BasicFeatureBuilder() or AdvancedFeatureBuilder() added (add_feature_engineering=True)
├── 4. SkewnessCorrector() added (add_skewness_correction=True)
├── 5. DataTypeOptimizer() added (optimize_dtypes=True)
├── 6. build_preprocessor() called (model_type='linear')
│   └── OneHotEncoder + StandardScaler  (Same as Linear)
├── 7. build_feature_selector() called (selector_type='lasso')
│   └── SelectFromModel(LassoCV)  ← Different!
├── 8. Ridge(alpha=1.0) added
└── 9. TransformedTargetRegressor() wrapping (target_log=True)
```

**Final Pipeline Structure:**
```
TransformedTargetRegressor(
  regressor=Pipeline([
    ('smart_impute', SmartImputer),
    ('outlier_handler', StatisticalOutlierHandler),
    ('feature_eng', BasicFeatureBuilder | AdvancedFeatureBuilder),
    ('skew_correct', SkewnessCorrector),
    ('dtype_opt', DataTypeOptimizer),
    ('preprocess', ColumnTransformer[OneHot+StandardScaler]),
    ('select', SelectFromModel[LassoCV]),  ← Different!
    ('model', Ridge)
  ]),
  func=log1p, inverse_func=expm1
)
```

### 3. Random Forest Model

**Call:**
```python
pipeline = make_pipeline(
    RandomForestRegressor(n_estimators=100, random_state=42), 
    **get_rf_config()
)
```

**Configuration (`get_rf_config`)**:
```python
{
    'model_type': 'tree',              # ← Different!
    'add_feature_engineering': True,
    'add_skewness_correction': False,  # ← Different! (RF handles skewness well)
    'handle_outliers': True,           # ← Still beneficial for RF
    'optimize_dtypes': True,
    'scale_numeric': False,            # ← Different! (RF doesn't need scaling)
    'selector_type': None,             # ← Different! (RF does implicit feature selection)
    'selector_kwargs': None,
    'target_log': True
}
```

**Function Call Flow:**
```
make_pipeline()
├── 1. SmartImputer() added (intelligent imputation)
├── 2. StatisticalOutlierHandler() added (handle_outliers=True)
├── 3. BasicFeatureBuilder() or AdvancedFeatureBuilder() added (add_feature_engineering=True)
├── 4. SkewnessCorrector() skipped (add_skewness_correction=False)
├── 5. DataTypeOptimizer() added (optimize_dtypes=True)
├── 6. build_tree_preprocessor() called (model_type='tree')
│   └── MultiLabelEncoder (no scaling)
├── 7. build_feature_selector() skipped (selector_type=None)
├── 8. RandomForestRegressor() added
└── 9. TransformedTargetRegressor() wrapping (target_log=True)
```

**Final Pipeline Structure:**
```
TransformedTargetRegressor(
  regressor=Pipeline([
    ('smart_impute', SmartImputer),
    ('outlier_handler', StatisticalOutlierHandler),
    ('feature_eng', BasicFeatureBuilder | AdvancedFeatureBuilder),
    ('dtype_opt', DataTypeOptimizer),
    ('preprocess', ColumnTransformer[MultiLabelEncoder]),
    ('model', RandomForestRegressor)
  ]),
  func=log1p, inverse_func=expm1
)
```

## Model Comparison Summary

| Step | Linear | Ridge | Random Forest |
|------|--------|-------|---------------|
| **Smart Imputation** | SmartImputer | SmartImputer | SmartImputer |
| **Outlier Handling** | StatisticalOutlierHandler | StatisticalOutlierHandler | StatisticalOutlierHandler |
| **Feature Engineering** | Basic/AdvancedFeatureBuilder | Basic/AdvancedFeatureBuilder | Basic/AdvancedFeatureBuilder |
| **Skewness Correction** | SkewnessCorrector | SkewnessCorrector | None |
| **Data Type Optimization** | DataTypeOptimizer | DataTypeOptimizer | DataTypeOptimizer |
| **Preprocessing** | OneHot + Scale | OneHot + Scale | Label only |
| **Feature Selection** | SelectKBest(k=20) | LassoCV | None |
| **Model** | LinearRegression | Ridge | RandomForest |
| **Target Transform** | log1p/expm1 | log1p/expm1 | log1p/expm1 |

## Key Differences

### 1. Linear vs Ridge
- **Similarity**: Same comprehensive preprocessing pipeline (imputation + outlier handling + feature engineering + skewness correction + encoding + scaling)
- **Difference**: Feature selection method only
  - Linear: SelectKBest with k=20 features
  - Ridge: SelectFromModel with LassoCV (threshold='median')

### 2. Linear/Ridge vs Random Forest
- **Shared preprocessing**: All models now share intelligent imputation, outlier handling, and data type optimization
- **Differences**:
  - **Skewness correction**: Linear/Ridge use it, RF skips it (robust to skewness)
  - **Encoding**: Linear models use OneHot, RF uses Label encoding
  - **Scaling**: Linear models scale features, RF doesn't need scaling
  - **Feature selection**: Linear models use explicit selection, RF has built-in importance
- **Pipeline complexity**: 
  - Linear/Ridge: 9 steps (comprehensive)
  - Random Forest: 6 steps (streamlined for tree-based models)

### 3. Enhanced Processing Benefits
- **Smart Imputation**: LotFrontage uses neighborhood-based median, improving accuracy
- **Statistical Outlier Handling**: IQR-based winsorization preserves data while reducing noise
- **Data Type Optimization**: Memory efficiency and categorical handling
- **Enhanced Skewness Correction**: Now includes outlier handling before transformation

### 4. Why These Differences?
- **OneHot vs Label encoding**: Linear models need numeric features in proper scale, tree models can handle categorical data directly
- **Scaling**: Linear models are sensitive to feature scale, tree models are not
- **Feature selection**: Linear models benefit from dimensionality reduction, tree models have built-in feature importance
- **Skewness correction**: Linear models assume normal distribution, tree models are robust to skewness
- **Outlier handling**: All models benefit from outlier management, but the approach differs (statistical winsorization vs tree robustness)

## Usage Recommendations

### For Linear Models (Linear, Ridge)
```python
preprocessor = build_preprocessor(model_type='linear', scale_numeric=True)
```

### For Tree Models (RandomForest, XGBoost)
```python
preprocessor = build_tree_preprocessor(scale_numeric=False)  # Safer encoding
```

### General Purpose
```python
preprocessor = build_preprocessor(model_type='linear'/'tree')  # Controlled by parameter
```

## Implementation Notes

### Two-Stage Architecture
- **Stage 1 (domain_data_cleaning.py)**: One-time domain-specific cleaning with data quality fixes
- **Stage 2 (pipelines.py)**: CV-safe statistical preprocessing within machine learning pipeline

### Enhanced Features
- **Smart Imputation**: Combines statistical methods with domain knowledge (neighborhood-based for LotFrontage)
- **Statistical Outlier Handling**: IQR-based winsorization (3.0 multiplier) preserves data while reducing noise
- **Enhanced Skewness Correction**: Now includes outlier handling before transformation selection
- **Data Type Optimization**: Automatic conversion of categorical numerics for memory efficiency
- **CV-Safe Processing**: All statistical operations are fit on training data and applied to validation data

### Safety Features
- `MultiLabelEncoder` safely handles unseen categories in tree models
- All preprocessing steps are applied within cross-validation folds to prevent data leakage
- Statistical parameters are learned from training data only during CV

## File Locations
This documentation corresponds to the implementation in:
- `/scripts/domain_data_cleaning.py` - Domain-specific data cleaning (Stage 1)
- `/scripts/pipelines.py` - Enhanced statistical pipeline (Stage 2)
- `/notebooks/03_basic_models/` - Model-specific usage examples

## Usage Workflow
```
1. Split Train Data → domain_data_cleaning.py → Domain-cleaned Data
2. Domain-cleaned Data → make_pipeline() → Model-ready Data → Training
```
