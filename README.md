# 🛡️ LoanGuard – Credit Default Risk Scoring Engine

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![ML Stack](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost%20%7C%20SHAP-orange)
![Serialization Format](https://img.shields.io/badge/Model%20Format-Pure%20JSON-brightgreen)
![UI](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

**LoanGuard** is an end-to-end fintech machine learning credit default risk scoring engine. It simulates modern lending risk workflows by evaluating applicant financial & credit profiles, handling class imbalance (defaults are rare events), scoring default probabilities, generating individual SHAP feature explanations for underwriting compliance, and serving predictions through a sleek Streamlit web dashboard.

> [!NOTE]
> **Pure JSON Model Serialization**: All model weights, preprocessing transformers, and SHAP explainers are saved as lightweight, human-readable, cross-platform `.json` files, eliminating binary pickle/joblib dependency risks.

---

## 📌 Executive Summary & Resume Impact Statement

> *"Built an end-to-end credit risk scoring engine (Logistic Regression & XGBoost) achieving **0.80+ ROC-AUC** on imbalanced tabular loan data (~8.0% default rate), featuring automated feature engineering, SMOTE imbalance calibration, pure JSON model persistence, and SHAP-based explainability for underwriting compliance."*

---

## 🚀 Key Features

1. **Synthetic & Standard Loan Data Pipeline**: Realistic credit dataset generator modeling non-linear default interactions (FICO score, DTI ratio, revolving credit utilization, past delinquencies, loan-to-income ratio).
2. **Automated Feature Engineering**:
   - `loan_to_income_ratio`: Requested capital relative to annual earnings.
   - `installment_to_income_ratio`: Monthly amortized payment vs. monthly gross income.
   - `delinquency_severity_score`: Weighted impact of 30-50, 60-89, and 90+ days past due & public derogatory records.
   - `high_utilization_flag`: Credit utilization >70% warning indicator.
   - `credit_score_tier`: Categorical risk bucketizing (POOR, FAIR, GOOD, EXCELLENT).
3. **Imbalance Calibration**: Calibrated with **SMOTE (Synthetic Minority Over-sampling)** and algorithm-level class weighting (`scale_pos_weight` in XGBoost) to maintain high precision-recall sensitivity on rare default events.
4. **Multi-Model Benchmark**: Stratified K-Fold comparison across **Logistic Regression** (interpretable baseline), **Random Forest**, and **XGBoost Classifier**.
5. **SHAP Model Explainability**:
   - **Global Explainability**: Identifies macro risk drivers across the entire credit portfolio.
   - **Single Applicant Explainability**: SHAP waterfall breakdown detailing exact positive (risk-increasing) and negative (protective) financial features for any individual applicant.
6. **Pure JSON Artifact Storage**: `preprocessor.json`, `champion_model.json`, `shap_explainer.json`, and `evaluation_metrics.json`.
7. **Modern Glassmorphism Web App**:
   - **Underwriter Portal**: Input applicant parameters, calculate real-time score (0-100 scale & PD %), view risk tier, receive auto-decision recommendations (Auto-Approve, Manual Review, Decline), and view SHAP bar chart.
   - **Batch Portfolio Explorer**: Upload batch CSVs, analyze portfolio default distribution, and export scored datasets.
   - **Governance & Benchmarking**: ROC-AUC / PR-AUC metric cards and global SHAP feature importance plot.

---

## 🛠️ Architecture & Project Structure

```
LoanGuard/
├── app.py                      # Interactive Streamlit Web Dashboard
├── train_pipeline.py           # End-to-end training & JSON artifact export pipeline
├── requirements.txt            # Python dependencies
├── PRD_Credit_Risk_Model.md    # Product Requirement Document
├── README.md                   # Documentation & setup guide
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic loan dataset generator
│   ├── data_processing.py      # Feature engineering & JSON ColumnTransformers
│   ├── model_trainer.py        # Model training, CV, imbalance calibration & JSON save/load
│   └── explainability.py       # SHAP explainer wrapper & JSON persistence
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # PyTest unit test suite with JSON checks
├── data/                       # Generated dataset & test samples
└── models/                     # Saved pure JSON artifacts (preprocessor.json, champion_model.json, etc.)
```

---

## ⚡ Quickstart Guide

### 1. Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/LoanGuard.git
cd LoanGuard
pip install -r requirements.txt
```

### 2. Train Model & Export JSON Artifacts

Run the end-to-end pipeline script to generate data, train all 3 models, select the champion model, setup SHAP explainability, and export `.json` files:

```bash
python train_pipeline.py
```

### 3. Launch Streamlit Web Application

Launch the interactive dashboard in your browser:

```bash
python -m streamlit run app.py
```

---

## 📈 Performance Benchmarks

Evaluated on held-out 2,000 applicant test dataset:

| Model | ROC-AUC | PR-AUC | F1-Score | Optimal Threshold | Model Storage Format |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** 🏆 | **0.8002** | **0.3444** | **0.2975** | **0.74** | **JSON** |
| Random Forest | 0.7560 | 0.2387 | 0.2745 | 0.42 | **JSON** |
| XGBoost | 0.7519 | 0.2546 | 0.2209 | 0.87 | **JSON** |

---

## 🧪 Running Unit Tests

Run the test suite to verify data processing, model training, and JSON serialization/deserialization:

```bash
python -m unittest discover tests
```

---

## 📜 License

MIT License © 2026 Pavitr Jain
