"""
Feature Engineering Classes for Ames Housing Dataset.
Provides minimal and comprehensive feature builders for model comparison.

Classes:
- BasicFeatureBuilder: Minimal feature engineering
- AdvancedFeatureBuilder: Comprehensive feature engineering
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin


class BasicFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Minimal feature engineering with only essential features.
    """
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self
        
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


class AdvancedFeatureBuilder(BasicFeatureBuilder):
    """
    Comprehensive feature engineering with interaction, ratio, derived, and categorical features.
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__(eps)
        self.neighborhood_mapping_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        super().fit(X, y)
        
        # Learn neighborhood categories based on target
        if y is not None and 'Neighborhood' in X.columns:
            temp_df = pd.DataFrame({
                'Neighborhood': X['Neighborhood'],
                'target': y
            })
            neighborhood_median = temp_df.groupby('Neighborhood')['target'].median()
            q1, q3 = neighborhood_median.quantile([0.25, 0.75])
            self.neighborhood_mapping_ = pd.cut(
                neighborhood_median,
                bins=[-np.inf, q1, q3, np.inf],
                labels=['Low', 'Medium', 'High']
            ).to_dict()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)
        eps = self.eps
        
        # Interaction features
        if {'OverallQual', 'GrLivArea'}.issubset(X.columns):
            X['Quality_x_Area'] = X['OverallQual'].astype(float) * X['GrLivArea']
            
        if {'OverallQual', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'}.issubset(X.columns):
            total_sf = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
            X['Quality_x_TotalSF'] = X['OverallQual'].astype(float) * total_sf
            
        if {'FullBath', 'HalfBath', 'GrLivArea'}.issubset(X.columns):
            total_bath = X['FullBath'] + 0.5 * X['HalfBath']
            X['Bath_x_Area'] = total_bath * X['GrLivArea']
            
        if {'GarageCars', 'OverallQual'}.issubset(X.columns):
            X['Garage_x_Quality'] = X['GarageCars'] * X['OverallQual'].astype(float)
        
        # Ratio features
        if {'GrLivArea', 'LotArea'}.issubset(X.columns):
            X['GrLivArea_to_LotArea'] = X['GrLivArea'] / (X['LotArea'] + eps)
            
        if {'1stFlrSF', 'GrLivArea'}.issubset(X.columns):
            X['1stFlr_to_GrLivArea'] = X['1stFlrSF'] / (X['GrLivArea'] + eps)
            
        if {'GarageArea', 'GrLivArea'}.issubset(X.columns):
            X['GarageArea_to_GrLivArea'] = X['GarageArea'] / (X['GrLivArea'] + eps)
            
        if {'TotalBsmtSF', 'GrLivArea'}.issubset(X.columns):
            X['BsmtArea_to_GrLivArea'] = X['TotalBsmtSF'] / (X['GrLivArea'] + eps)
            
        if {'GrLivArea', 'TotRmsAbvGrd'}.issubset(X.columns):
            X['AvgRoomSize'] = X['GrLivArea'] / (X['TotRmsAbvGrd'] + eps)
        
        # Replace infinite values with 0
        ratio_cols = [col for col in X.columns if '_to_' in col or 'AvgRoomSize' in col]
        for col in ratio_cols:
            if col in X.columns:
                X[col] = X[col].replace([np.inf, -np.inf], 0)
        
        # Derived features
        if {'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'}.issubset(X.columns):
            X['TotalBathrooms'] = (
                X['FullBath'] + 0.5 * X['HalfBath'] + 
                X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath']
            )
            
        if {'YrSold', 'YearBuilt', 'YearRemodAdd'}.issubset(X.columns):
            X['EffectiveAge'] = X['YrSold'] - X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
            
        porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        available_porch_cols = [col for col in porch_cols if col in X.columns]
        if available_porch_cols:
            X['TotalPorchSF'] = X[available_porch_cols].sum(axis=1)
        
        # Categorical features
        if 'HouseAge' in X.columns:
            X['AgeCategory'] = pd.cut(
                X['HouseAge'],
                bins=[0, 10, 30, 50, 200],
                labels=['New', 'Mid', 'Old', 'Very_Old'],
                include_lowest=True
            ).astype('category')
            
        if {'YearBuilt', 'YrSold'}.issubset(X.columns):
            X['IsNew'] = (X['YearBuilt'] == X['YrSold']).astype('category')
            
        if '2ndFlrSF' in X.columns:
            X['Has2ndFloor'] = (X['2ndFlrSF'] > 0).astype('category')
            
        if 'TotalBsmtSF' in X.columns:
            X['HasBasement'] = (X['TotalBsmtSF'] > 0).astype('category')
            
        if 'Fireplaces' in X.columns:
            X['HasFireplace'] = (X['Fireplaces'] > 0).astype('category')
            
        if 'PoolArea' in X.columns:
            X['HasPool'] = (X['PoolArea'] > 0).astype('category')
            
        if 'MasVnrArea' in X.columns:
            X['HasMasVnrArea'] = (X['MasVnrArea'] > 0).astype('category')
            
        if 'TotalPorchSF' in X.columns:
            X['HasPorch'] = (X['TotalPorchSF'] > 0).astype('category')
            
        if 'OverallQual' in X.columns:
            X['QualityCategory'] = pd.cut(
                X['OverallQual'].astype(float),
                bins=[0, 4, 7, 10],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            ).astype('category')
            
        if 'Neighborhood' in X.columns and self.neighborhood_mapping_ is not None:
            X['NeighborhoodCategory'] = X['Neighborhood'].map(self.neighborhood_mapping_).astype('category')
        
        return X
