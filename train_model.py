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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from category_encoders import TargetEncoder
import featuretools as ft

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def train_and_export():
    print("=" * 70)
    print("🚀 PSO-ML20 HOUSE PRICE VALUATION MODEL TRAINING ENGINE")
    print("=" * 70)
    
    data_paths = [
        'house_sales.csv',
        r'C:\Users\Israel\Desktop\PROJECTS\House-Pricing-Project\house_sales.csv',
        r'C:\Users\Israel\Downloads\HOUSE_PRICE_DATA.csv'
    ]
    
    df_raw = None
    for p in data_paths:
        if os.path.exists(p):
            print(f"📁 Loading dataset from: {p}")
            df_raw = pd.read_csv(p)
            break
            
    if df_raw is None:
        raise FileNotFoundError("Could not find house sales CSV file!")
        
    TARGET_COL = 'SalePrice' if 'SalePrice' in df_raw.columns else 'price'
    
    df = df_raw.copy()
    cols_to_remove = ['Serial_Number', 'Full_Name', 'Description', 'id', 'PropertyID', 'AdjSalePrice']
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore').drop_duplicates().reset_index(drop=True)
    df.columns = df.columns.str.strip()

    # Create manual features
    df['house_age'] = datetime.now().year - pd.to_numeric(df['YrBuilt'], errors='coerce').fillna(1980)
    
    for date_col in ['DocumentDate', 'date', 'Date']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['DocumentDate_year'] = df[date_col].dt.year
            df['DocumentDate_month'] = df[date_col].dt.month
            df['DocumentDate_weekday'] = df[date_col].dt.weekday
            df['DocumentDate_is_weekend'] = df[date_col].dt.weekday.isin([5, 6]).astype(int)
            df = df.drop(columns=[date_col])

    # Target hardening
    df = df[df[TARGET_COL] > 0]
    upper_limit = df[TARGET_COL].quantile(0.99)
    df = df[df[TARGET_COL] <= upper_limit].reset_index(drop=True)

    print("⛏️ Executing FeatureTools Deep Feature Synthesis (DFS)...")
    X_temp = df.drop(columns=[TARGET_COL], errors='ignore').select_dtypes(include=[np.number]).fillna(-999)
    y_temp = df[TARGET_COL]

    filter_model = RandomForestRegressor(n_estimators=50, max_depth=1, n_jobs=-1, random_state=42)
    filter_model.fit(X_temp, y_temp)
    importance = pd.Series(filter_model.feature_importances_, index=X_temp.columns)
    top_building_blocks = importance.nlargest(200).index.tolist()
    base_cols = [c for c in top_building_blocks if '_squared' not in c and '_per_' not in c]

    es = ft.EntitySet(id='automated_mining')
    df_ft = df[base_cols].reset_index()
    es = es.add_dataframe(dataframe_name='main_table', dataframe=df_ft, index='index')

    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name='main_table',
        trans_primitives=['add_numeric', 'multiply_numeric'],
        max_depth=1,
        verbose=False
    )
    df_new = feature_matrix.drop(columns=['index'], errors='ignore')
    df = pd.concat([df, df_new], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    print("⚖️ Strict Judge: Selecting Top 45 Predictive Signals...")
    X = df.drop(columns=[TARGET_COL], errors='ignore')
    y = df[TARGET_COL]

    X_for_judge = X.copy()
    for col in X_for_judge.columns:
        if X_for_judge[col].dtype == 'object' or X_for_judge[col].dtype.name == 'category':
            X_for_judge[col] = X_for_judge[col].astype('category').cat.codes
        X_for_judge[col] = X_for_judge[col].replace([np.inf, -np.inf], np.nan).fillna(-999)

    model_judge = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model_judge.fit(X_for_judge, y)

    importance_df = pd.DataFrame({'feature': X_for_judge.columns, 'importance': model_judge.feature_importances_}).sort_values(by='importance', ascending=False)
    top_45 = importance_df['feature'].head(45).tolist()

    df_final = df[top_45 + [TARGET_COL]]

    X_final = df_final.drop(columns=[TARGET_COL])
    y_final = df_final[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

    num_df = X_final.select_dtypes(include='number')
    cat_df = X_final.select_dtypes(exclude='number')

    power_cols = num_df.columns[num_df.skew().abs() > 0.75].tolist()
    standard_cols = num_df.columns[num_df.skew().abs() <= 0.75].tolist()

    high_nom = cat_df.columns[cat_df.nunique() > 30].tolist()
    low_nom = cat_df.columns[cat_df.nunique() <= 30].tolist()

    transformers = []
    if power_cols:
        transformers.append(('pow', Pipeline([('im', SimpleImputer(strategy='median')), ('pt', PowerTransformer(method='yeo-johnson')), ('ss', StandardScaler())]), power_cols))
    if standard_cols:
        transformers.append(('std', Pipeline([('im', SimpleImputer(strategy='mean')), ('ss', StandardScaler())]), standard_cols))
    if high_nom:
        transformers.append(('hi', Pipeline([('im', SimpleImputer(strategy='constant', fill_value='Missing')), ('te', TargetEncoder())]), high_nom))
    if low_nom:
        transformers.append(('lo', Pipeline([('im', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), low_nom))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    cb = CatBoostRegressor(n_estimators=2500, learning_rate=0.02, depth=6, l2_leaf_reg=15, random_seed=42, verbose=0, thread_count=-1)
    model = TransformedTargetRegressor(regressor=cb, func=np.log1p, inverse_func=np.expm1)

    full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    print("⏳ Fitting CatBoost Pipeline on df_final...")
    full_pipeline.fit(X_train, y_train)

    preds = full_pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)

    print("\n" + "=" * 60)
    print("💎 MODEL INTEGRITY CERTIFICATE (EXACT MATCH)")
    print("=" * 60)
    print(f"🎯 R² Score (Accuracy) : {r2:.4f} ({r2*100:.2f}%)")
    print(f"💵 MAE (Mean Abs Error): ${mae:,.2f}")
    print(f"📐 RMSE                : ${rmse:,.2f}")
    print(f"📉 MAPE (%)            : {mape*100:.2f}%")
    print("=" * 60)

    top_features_dict = importance_df.head(15).set_index('feature')['importance'].to_dict()

    model_bundle = {
        'pipeline': full_pipeline,
        'feature_columns': X_final.columns.tolist(),
        'top_45_features': top_45,
        'base_raw_columns': X_temp.columns.tolist(),
        'target_col': TARGET_COL,
        'metrics': {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        },
        'top_features': top_features_dict,
        'sample_input': X_final.head(5).to_dict(orient='records'),
        'trained_at': datetime.now().isoformat()
    }

    joblib.dump(model_bundle, 'house_price_model.joblib')
    print("✅ Successfully saved high-accuracy 'house_price_model.joblib'!")

if __name__ == '__main__':
    train_and_export()
