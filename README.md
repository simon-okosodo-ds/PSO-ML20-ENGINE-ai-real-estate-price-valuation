# 🏠 PSO-ML20 Industrial House Price Valuation Engine & Web UI

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-FFCC00.svg)](https://catboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade **Real Estate Valuation Web Application** powered by an advanced Gradient Boosting Machine Learning ensemble (`CatBoost` / `XGBoost`). The system features automated feature engineering, target log-transformations, batch CSV file processing with predictions download, single-property real-time valuation, and an interactive model integrity dashboard.

---

## 🌟 Key Features

- 🏡 **Single Property Real-Time Valuation Calculator**: Input property features (Bedrooms, Bathrooms, SqFt Living, Lot Size, Building Grade, Zipcode, Year Built) to obtain instant estimated market price ($), price per sqft ($/sqft), and 95% valuation confidence bounds.
- 📊 **Batch CSV Upload Engine**: Drag and drop custom housing CSV datasets to run bulk automated valuations. View interactive results tables and price distribution charts.
- 📥 **1-Click Predictions Download**: Export batch valuation results directly into formatted CSV files.
- 🧪 **Sample CSV Template**: Download pre-formatted sample input CSV files for instant testing.
- 📈 **Model Integrity & Audit Dashboard**: View certified model metrics ($R^2$, MAE, RMSE, MAPE) and strategic signal hierarchy (feature importances).
- 🎨 **Executive Obsidian UI Design**: Built with a sleek dark slate glassmorphism theme, metric cards, and responsive components.

---

## 💎 Model Integrity Certificate

| Metric | Certified Score | Status |
| :--- | :--- | :--- |
| **R² Score (Accuracy)** | **88.4% - 91.2%** | 🎯 `CERTIFIED ELITE` |
| **MAE (Mean Absolute Error)** | **~$52,000** | ✅ `ROCK SOLID` |
| **Target Pipeline** | `TransformedTargetRegressor` (Log1p / Expm1) | 🛡️ `STABLE` |
| **Primary Estimator** | `CatBoostRegressor` / `HistGradientBoostingRegressor` | 🚀 `PRODUCTION READY` |

---

## 📁 Repository Structure

```text
├── app.py                                   # Streamlit Web Application UI
├── train_model.py                           # ML Training, Feature Engineering & Export Script
├── house_price_model.joblib                 # Exported Production Model Bundle
├── Regression_Project_STABLE_House_pricing.ipynb # Original Research & Audit Notebook
├── house_sales.csv                          # Housing Dataset
├── requirements.txt                         # Python Dependencies
├── .gitignore                               # Git Ignore Configuration
└── README.md                                # System Documentation
```

---

## 🚀 Quickstart - Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/simon-okosodo-ds/Pso-ml20-interfaces.git
cd Pso-ml20-interfaces
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model Pipeline (Optional - Pre-bundled)
```bash
python train_model.py
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
*The web interface will open automatically in your default browser at `http://localhost:8501`.*

---

## 🌐 Free 1-Click Public Deployment (Streamlit Community Cloud)

To share this application live with anyone using a public shareable URL:

1. Push this repository to your GitHub account (`https://github.com/simon-okosodo-ds`).
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with your GitHub account.
4. Click **"New app"** and select:
   - **Repository**: `simon-okosodo-ds/Pso-ml20-interfaces`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**
6. Within ~60 seconds, your app will be live with a permanent URL such as:
   `https://house-price-valuation.streamlit.app`

---

## 📋 CSV Upload Schema Specification

For batch CSV uploads, the uploaded CSV can contain any of the following standard column names:

- `SqFtTotLiving` / `sqft_living` (Number): Total living area in square feet
- `Bedrooms` / `bedrooms` (Number): Number of bedrooms
- `Bathrooms` / `bathrooms` (Number): Number of bathrooms
- `SqFtLot` / `sqft_lot` (Number): Total lot size in square feet
- `YrBuilt` / `yr_built` (Number): Construction year (e.g., 2005)
- `BldgGrade` (Number): Building quality grade (1 to 13)
- `ZipCode` (Number/String): Property postal code
- `PropertyType` (String): e.g. Single Family, Townhouse, Condo
- `DocumentDate` / `date` (Date): Sale date (YYYY-MM-DD)

---

## 👤 Author & Maintainer

- **Developer**: Simon Okosodo
- **GitHub**: [simon-okosodo-ds](https://github.com/simon-okosodo-ds)
- **Project**: PSO-ML20 Machine Learning Interfaces
