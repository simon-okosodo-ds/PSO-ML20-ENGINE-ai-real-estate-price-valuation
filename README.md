# 🏠 PSO-ML20 Industrial House Price Valuation Engine & Web UI

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-FFCC00.svg)](https://catboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade **Real Estate Valuation Web Application** powered by an advanced Gradient Boosting Machine Learning ensemble (`CatBoost`, `XGBoost`, `LightGBM`). The system features automated feature engineering, target log-transformations, batch CSV file processing with predictions download, single-property real-time valuation, and an interactive model integrity dashboard.

---

## 🏆 Championship Tournament Leaderboard (From Notebook Audition)

```text
MODEL           | R2       | MAE          | RMSE         | MAPE    
---------------------------------------------------------------------------
LightGBM_Reg    | 0.8813   | $45,242      | $69,510      | 10.82%  
XGBoost_Reg     | 0.8811   | $44,860      | $69,567      | 10.78%  
CatBoost_Reg    | 0.8796   | $45,487      | $70,012      | 10.89%  
```

---

## 🌟 Key Features

- 🏡 **Single Property Real-Time Valuation Calculator**: Input property features (Bedrooms, Bathrooms, SqFt Living, Lot Size, Building Grade, Zipcode, Year Built, LandVal, ImpsVal) to obtain instant estimated market price ($), price per sqft ($/sqft), and 95% valuation confidence bounds.
- 📊 **Batch CSV Upload Engine**: Drag and drop custom housing CSV datasets to run bulk automated valuations. View interactive results tables and price distribution charts.
- 📥 **1-Click Predictions Download**: Export batch valuation results directly into formatted CSV files.
- 🧪 **Sample CSV Template**: Download pre-formatted sample input CSV files for instant testing.
- 📈 **Model Integrity & Audit Dashboard**: View certified model metrics ($R^2$, MAE, RMSE, MAPE) and strategic signal hierarchy (feature importances).
- 🎨 **Executive Obsidian UI Design**: Built with a sleek dark slate glassmorphism theme, metric cards, and responsive components.

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

### 3. Launch the Web Application
```bash
streamlit run app.py
```
*The web interface will open automatically in your default browser at `http://localhost:8501`.*

---

## 🌐 Free 1-Click Public Deployment (Streamlit Community Cloud)

To share this application live with anyone using a public shareable URL:

1. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with your GitHub account (**simon-okosodo-ds**).
3. Click **"New app"** and select:
   - **Repository**: `simon-okosodo-ds/Pso-ml20-interfaces`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"**

---

## 👤 Author & Maintainer

- **Developer**: Simon Okosodo
- **GitHub**: [simon-okosodo-ds](https://github.com/simon-okosodo-ds)
- **Project**: PSO-ML20 Machine Learning Interfaces
