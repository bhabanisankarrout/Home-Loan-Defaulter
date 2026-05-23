# 🏠 Home Loan Defaulter Prediction

A machine learning project to identify customers at risk of defaulting on home loans, enabling financial institutions to make data-driven lending decisions while promoting financial inclusion.

---

## 📌 Problem Statement

Predict whether a loan applicant will default on their home loan using historical credit data, behavioral patterns, and financial indicators. The goal is to help lenders minimize default risk without unfairly excluding creditworthy individuals who may have limited credit history.

---

## 📂 Dataset

This project uses the **Home Credit Default Risk** dataset, which consists of 7 relational tables:

| File | Description |
|---|---|
| `application_train.csv` | Main table with applicant features and the target variable (`TARGET`) |
| `bureau.csv` | Applicant's credit history from the Credit Bureau |
| `bureau_balance.csv` | Monthly balance snapshots of bureau credits |
| `previous_application.csv` | Previous loan applications at Home Credit |
| `POS_CASH_balance.csv` | Monthly POS and cash loan balance history |
| `credit_card_balance.csv` | Monthly credit card balance history |
| `installments_payments.csv` | Installment payment history |

> **Target Variable:** `TARGET` — `1` = Payment difficulties (defaulter), `0` = All payments on time

---

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed comprehensive EDA across all 7 datasets:

- **Univariate analysis** — distributions of numerical and categorical features
- **Bivariate analysis** — feature relationships with the target variable
- **Correlation heatmaps** — identifying multicollinearity and redundant features

Key findings:
- The dataset has significant **class imbalance** (~8% defaulters)
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` are the strongest predictors
- `DAYS_BIRTH` (age) shows clear separation — younger applicants default more
- Several building/property `_AVG`, `_MODE`, `_MEDI` features have low predictive power
- Placeholder value `365243` used for missing dates in multiple columns

### 2. Data Preprocessing

- **Undersampling** — balanced dataset to 100,000 samples (all defaulters + sampled non-defaulters)
- **Relational filtering** — synced auxiliary tables to match the sampled applicant IDs
- **Outlier capping** — clipped numerical features at the 1st and 99th percentiles
- **Missing value imputation** — median for numerical, mode for categorical features
- **Encoding** — Label Encoding for categorical variables
- **Scaling** — RobustScaler to handle outliers

### 3. Feature Engineering

Generated domain-driven features from each table:

**Application features:**
- Age in years, age group
- Employment-to-age ratio
- Income-to-credit and income-to-annuity ratios
- External source mean, std, min, max
- Document submission count

**Bureau features:**
- Credit count, overdue history, debt-to-credit ratio, overdue ratio

**Previous application features:**
- Historical approval rate, consumer loan count, application count

**POS Cash & Credit Card features:**
- DPD aggregates, balance-to-limit ratio, drawing patterns

**Installment features:**
- Payment difference, late payment flag, payment ratio

### 4. Feature Selection

- Used **Mutual Information** to rank features
- Selected top **100 features** out of the full merged feature set
- Applied **PCA** for dimensionality analysis (95% variance explained by ~60 components)

### 5. Model Training & Evaluation

Trained and compared 6 models:

| Model | ROC-AUC |
|---|---|
| **LightGBM** | **0.785** ✅ |
| Gradient Boosting | 0.780 |
| XGBoost | 0.773 |
| Logistic Regression | 0.769 |
| Random Forest | 0.762 |
| Decision Tree | 0.700 |

All models used class-imbalance handling (`class_weight='balanced'` or `scale_pos_weight`).

### 6. Hyperparameter Tuning

Used **RandomizedSearchCV** (50 iterations, 5-fold CV) on the top 4 models:
- Random Forest
- Gradient Boosting
- LightGBM
- XGBoost

### 7. Customer Risk Segmentation

Segmented predictions into three tiers:

| Risk Segment | Default Rate |
|---|---|
| Low Risk | ~6.4% |
| Medium Risk | ~25% |
| High Risk | ~51.5% |

---

## 📊 Final Model Performance (LightGBM)

- **ROC-AUC:** 0.785
- **Overall Accuracy:** 72.3%
- **Defaulter Recall:** 68.8%
- **Weighted F1-Score:** 0.73

The model correctly identifies ~70% of actual defaulters, enabling the institution to focus manual review on high-risk applicants while automating approvals for low-risk segments.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Dimensionality Reduction:** `sklearn.decomposition.PCA`

---

## 📁 Project Structure

```
home-loan-defaulter-prediction/
│
├── data/
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── credit_card_balance.csv
│   └── installments_payments.csv
│
├── Home_Loan_Defaulter_Prediction.ipynb   # Main notebook
├── model_comparison_results.csv           # Model evaluation output
├── pca_scree_plot.png                     # PCA variance plot
├── pca_2d_scatter.png                     # PCA 2D class separation
├── pca_loadings.png                       # PCA feature loadings
└── README.md
```

---

## ⚠️ Key Challenges & Solutions

| Challenge | Solution |
|---|---|
| **7 relational tables** | Used `groupby` aggregation and merged statistical summaries into the main dataset |
| **High missing values** (e.g., 60%+ in some columns) | Median imputation; missing indicator flags where missingness carries information |
| **Class imbalance** (~8% defaulters) | Undersampling + `class_weight='balanced'` / `scale_pos_weight` |
| **Placeholder dates** (value `365243`) | Treated as missing/NaN before imputation |
| **Multicollinearity** in building features | PCA across `_AVG`, `_MODE`, `_MEDI` blocks; dropped redundant columns |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

### Run

1. Clone this repository
2. Place the dataset CSVs in the `data/` directory
3. Open and run `Home_Loan_Defaulter_Prediction.ipynb` in Jupyter

---

## 📈 Future Improvements

- Experiment with SMOTE oversampling instead of undersampling
- Add stacking/blending of top models
- Deploy the model as a REST API using Flask or FastAPI
- Add SHAP values for model explainability

---
