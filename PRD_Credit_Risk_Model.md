# PRD: Credit Risk / Loan Default Prediction Model

## 1. Overview
**Project Name:** LoanGuard – Credit Default Risk Scoring Engine
**Owner:** Pavitr Jain
**Type:** Personal data science project (fintech-focused)
**Tools:** Python, scikit-learn, XGBoost, SHAP, Streamlit, Antigravity (agentic IDE for scaffolding/pipeline code)

## 2. Problem Statement
Lenders (banks, NBFCs, fintech credit apps) need to estimate the probability that a loan applicant will default, so they can price risk correctly and make approve/reject decisions. The goal of this project is to replicate that workflow end-to-end: from raw applicant data to an explainable risk score.

## 3. Objective
Build a machine learning model that predicts loan default probability from applicant financial and demographic data, with a strong focus on:
- Handling realistic class imbalance (defaults are rare)
- Producing interpretable, explainable outputs (not just a black-box score)
- Evaluating performance the way a lender actually would (not just accuracy)

## 4. Target Users (Hypothetical)
- Underwriting/credit risk teams who need a quick risk score per applicant
- Compliance teams who need to justify why an applicant was flagged as high-risk

## 5. Data
- **Source:** Public dataset — Lending Club loan data or Kaggle "Give Me Some Credit"
- **Key fields:** income, debt-to-income ratio, credit utilization, number of open credit lines, delinquency history, loan amount, employment length
- **Target variable:** Default / non-default (binary)

## 6. Scope

### In Scope
- Data cleaning and feature engineering (DTI, utilization ratio, payment history features)
- Class imbalance handling (SMOTE, class weighting)
- Model training: Logistic Regression (baseline/interpretable), Random Forest, XGBoost
- Model explainability via SHAP (global + per-applicant)
- Evaluation via ROC-AUC, Precision-Recall AUC, F1 at business-relevant thresholds
- A simple Streamlit app: input applicant details → get risk score + top SHAP drivers

### Out of Scope
- Real-time scoring infrastructure / production deployment
- Actual regulatory compliance certification
- Live integration with a bank's core system

## 7. Functional Requirements
| ID | Requirement |
|----|-------------|
| FR1 | System shall ingest and clean raw applicant-level tabular data |
| FR2 | System shall engineer at least 5 derived risk features |
| FR3 | System shall train and compare ≥3 classification models |
| FR4 | System shall address class imbalance quantitatively (document before/after metrics) |
| FR5 | System shall generate SHAP explanations for individual predictions |
| FR6 | System shall expose a simple UI to score a new applicant on demand |

## 8. Success Metrics
- ROC-AUC ≥ 0.75 on held-out test set
- Precision-Recall AUC reported and compared across models (since classes are imbalanced)
- Clear SHAP-based explanation available for any given prediction
- Working demo app that returns a score in under 2 seconds

## 9. Milestones
| Phase | Deliverable | Target |
|-------|-------------|--------|
| 1 | Data cleaning + EDA notebook | Week 1 |
| 2 | Feature engineering + imbalance handling | Week 2 |
| 3 | Model training + comparison + tuning | Week 3 |
| 4 | SHAP explainability integration | Week 4 |
| 5 | Streamlit app + README + resume writeup | Week 5 |

## 10. Risks & Mitigations
- **Risk:** Dataset may not reflect real-world fintech data quality → Mitigate by clearly documenting assumptions and limitations in README
- **Risk:** Overfitting on imbalanced data → Mitigate with stratified k-fold CV and PR-AUC as primary metric, not accuracy
- **Risk:** SHAP compute cost on large models → Use TreeExplainer (fast for tree-based models) and subsample if needed

## 11. Resume Impact Statement
"Built a credit risk scoring model (XGBoost) achieving [X] ROC-AUC on imbalanced loan data, with SHAP-based explainability for underwriting decisions."
