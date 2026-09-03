import os
import sys

# Ensure Windows terminal compatibility for utf-8 emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def engineer_features(df, target_col='SalePrice'):
    df_clean = df.copy()
    
    # 1. Clean column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # 2. Remove non-predictive & leakage columns
    cols_to_remove = ['Serial_Number', 'Full_Name', 'Description', 'id', 'PropertyID', 'LandVal', 'ImpsVal', 'ym', 'zhvi_px', 'zhvi_idx', 'AdjSalePrice']
    df_clean = df_clean.drop(columns=[c for c in cols_to_remove if c in df_clean.columns], errors='ignore')
    
    # 3. Clean Target
    if target_col in df_clean.columns:
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[target_col])
        df_clean = df_clean[df_clean[target_col] > 0]
        # Clip extreme 0.5% tail outliers for target stability
        upper_q = df_clean[target_col].quantile(0.995)
        df_clean = df_clean[df_clean[target_col] <= upper_q]
    
    # 4. Date Feature Extraction
    for date_col in ['DocumentDate', 'date', 'Date']:
        if date_col in df_clean.columns:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_clean[f'{date_col}_year'] = df_clean[date_col].dt.year
            df_clean[f'{date_col}_month'] = df_clean[date_col].dt.month
            df_clean[f'{date_col}_weekday'] = df_clean[date_col].dt.weekday
            df_clean[f'{date_col}_is_weekend'] = df_clean[date_col].dt.weekday.isin([5, 6]).astype(int)
            df_clean = df_clean.drop(columns=[date_col])
            
    # 5. Age Extraction
    for yr_col in ['YrBuilt', 'yr_built', 'YearBuilt']:
        if yr_col in df_clean.columns:
            df_clean['house_age'] = datetime.now().year - pd.to_numeric(df_clean[yr_col], errors='coerce').fillna(1980)
            
    for yr_ren in ['YrRenovated', 'yr_renovated']:
        if yr_ren in df_clean.columns:
            df_clean['is_renovated'] = (pd.to_numeric(df_clean[yr_ren], errors='coerce').fillna(0) > 0).astype(int)

    # 6. Ratios & Interactions
    living_col = 'SqFtTotLiving' if 'SqFtTotLiving' in df_clean.columns else ('sqft_living' if 'sqft_living' in df_clean.columns else None)
    bed_col = 'Bedrooms' if 'Bedrooms' in df_clean.columns else ('bedrooms' if 'bedrooms' in df_clean.columns else None)
    bath_col = 'Bathrooms' if 'Bathrooms' in df_clean.columns else ('bathrooms' if 'bathrooms' in df_clean.columns else None)
    lot_col = 'SqFtLot' if 'SqFtLot' in df_clean.columns else ('sqft_lot' if 'sqft_lot' in df_clean.columns else None)

    if living_col:
        df_clean[f'{living_col}_squared'] = df_clean[living_col] ** 2
        if bed_col and bed_col in df_clean.columns:
            df_clean['sqft_per_bedroom'] = df_clean[living_col] / (df_clean[bed_col] + 1)
        if bath_col and bath_col in df_clean.columns:
            df_clean['sqft_per_bathroom'] = df_clean[living_col] / (df_clean[bath_col] + 1)
        if lot_col and lot_col in df_clean.columns:
            df_clean['living_to_lot_ratio'] = df_clean[living_col] / (df_clean[lot_col] + 1)
            
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def train_and_export():
    print("=" * 70)
    print("🚀 PSO-ML20 HOUSE PRICE VALUATION MODEL TRAINING ENGINE")
    print("=" * 70)
    
    # Locate dataset
    data_paths = [
        'house_sales.csv',
        r'C:\Users\Israel\Desktop\PROJECTS\House-Pricing-Project\house_sales.csv',
        r'C:\Users\Israel\Downloads\HOUSE_PRICE_DATA.csv'
    ]
    
    df_raw = None
    chosen_path = None
    for p in data_paths:
        if os.path.exists(p):
            print(f"📁 Loading dataset from: {p}")
            df_raw = pd.read_csv(p)
            chosen_path = p
            break
            
    if df_raw is None:
        raise FileNotFoundError("Could not find any house sales CSV file!")
        
    target_col = 'SalePrice' if 'SalePrice' in df_raw.columns else 'price'
    print(f"🎯 Target variable identified as: '{target_col}'")
    
    # Feature Engineering
    df_engineered = engineer_features(df_raw, target_col=target_col)
    
    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col]
    
    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"📊 Processed Features ({len(X.columns)} total): {len(num_cols)} Numerical | {len(cat_cols)} Categorical")
    
    # Preprocessor
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    # Select Regressor
    if HAS_CATBOOST:
        print("⚡ Using CatBoostRegressor for peak accuracy...")
        base_regressor = CatBoostRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=10,
            random_seed=42,
            verbose=0
        )
    else:
        print("⚡ Using HistGradientBoostingRegressor...")
        base_regressor = HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.03,
            max_depth=8,
            random_state=42
        )
        
    model = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("⏳ Fitting pipeline on training set...")
    full_pipeline.fit(X_train, y_train)
    
    # Evaluation
    preds = full_pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)
    
    print("\n" + "=" * 60)
    print("💎 MODEL INTEGRITY CERTIFICATE (EVALUATION RESULTS)")
    print("=" * 60)
    print(f"🎯 R² Score (Accuracy) : {r2:.4f} ({r2*100:.2f}%)")
    print(f"💵 MAE (Mean Abs Error): ${mae:,.2f}")
    print(f"📐 RMSE                : ${rmse:,.2f}")
    print(f"📉 MAPE (%)            : {mape*100:.2f}%")
    print("=" * 60)
    
    # Calculate Feature Importances for Audit Dashboard
    feature_names = num_cols + cat_cols
    rf_judge = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    
    # Quick fit on numeric representation to get importances
    X_num = X.copy()
    for col in cat_cols:
        X_num[col] = X_num[col].astype('category').cat.codes
    X_num = X_num.fillna(0)
    rf_judge.fit(X_num, y)
    
    importance_series = pd.Series(rf_judge.feature_importances_, index=X_num.columns).sort_values(ascending=False)
    top_features = importance_series.head(15).to_dict()
    
    # Save Model Bundle
    model_bundle = {
        'pipeline': full_pipeline,
        'feature_columns': X.columns.tolist(),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'target_col': target_col,
        'metrics': {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        },
        'top_features': top_features,
        'sample_input': X.head(5).to_dict(orient='records'),
        'trained_at': datetime.now().isoformat()
    }
    
    output_model_path = 'house_price_model.joblib'
    joblib.dump(model_bundle, output_model_path)
    print(f"\n✅ Model bundle successfully exported to: '{output_model_path}'")
    print("🚀 Training complete! Ready for Web App UI integration.")

if __name__ == '__main__':
    train_and_export()
