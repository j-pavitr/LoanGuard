import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import CreditDataPreprocessor
from src.model_trainer import load_model_from_json
from src.explainability import CreditRiskExplainer

# Set page config with custom title and wide layout
st.set_page_config(
    page_title="LoanGuard | Credit Default Risk Scoring Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Rich Dark Glassmorphism Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Header Gradient Banner */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #030712 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    /* Risk Badges */
    .badge-approve {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-review {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-decline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    models_dir = "models"
    data_dir = "data"
    
    if not os.path.exists(os.path.join(models_dir, "champion_model.json")):
        from train_pipeline import run_pipeline
        run_pipeline()
        
    preprocessor = CreditDataPreprocessor.from_json(os.path.join(models_dir, "preprocessor.json"))
    champion_model = load_model_from_json(os.path.join(models_dir, "champion_model.json"))
    shap_explainer = CreditRiskExplainer.from_json(
        champion_model, preprocessor.feature_names, os.path.join(models_dir, "shap_explainer.json")
    )
    
    with open(os.path.join(models_dir, "evaluation_metrics.json"), "r") as f:
        metrics = json.load(f)
        
    test_df = pd.read_csv(os.path.join(data_dir, "test_sample.csv"))
    
    return preprocessor, champion_model, shap_explainer, metrics, test_df


# Header Section
st.markdown("""
<div class="header-card">
    <div class="header-title">🛡️ LoanGuard – Credit Default Risk Scoring Engine</div>
    <div class="header-subtitle">
        AI-powered underwriting risk platform with default scoring, class-imbalance resilience, and SHAP explainability.
    </div>
</div>
""", unsafe_allow_html=True)

try:
    preprocessor, champion_model, shap_explainer, metrics, test_df = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run `train_pipeline.py` first.")
    st.stop()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "👤 Single Applicant Underwriting", 
    "📊 Batch Portfolio Analytics", 
    "⚙️ Model Performance & Governance"
])


# ==========================================
# TAB 1: SINGLE APPLICANT UNDERWRITING
# ==========================================
with tab1:
    st.markdown("### Applicant Profile & Loan Application")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.subheader("Financial Profile")
        annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=75000, step=2500)
        emp_length_years = st.slider("Employment Length (Years)", min_value=0, max_value=30, value=5)
        home_ownership = st.selectbox("Home Ownership Status", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=1)
        credit_score = st.slider("FICO / Credit Score", min_value=350, max_value=850, value=680)

    with col_input2:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=1000)
        loan_intent = st.selectbox("Loan Purpose", ["DEBTCONSOLIDATION", "PERSONAL", "HOMEIMPROVEMENT", "VENTURE", "MEDICAL", "EDUCATION"])
        loan_term_months = st.radio("Loan Term", [36, 60], index=0, horizontal=True)
        interest_rate = st.slider("Interest Rate (%)", min_value=4.0, max_value=30.0, value=11.5, step=0.25)

    with col_input3:
        st.subheader("Credit Line History")
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio (%)", min_value=0.0, max_value=60.0, value=22.0, step=0.5)
        revolving_utilization = st.slider("Revolving Credit Utilization (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
        total_open_acc = st.number_input("Total Open Credit Lines", min_value=1, max_value=40, value=9)
        delinq_2yrs = st.number_input("30+ Days Past Due (Last 2 Yrs)", min_value=0, max_value=10, value=0)
        derogatory_recs = st.number_input("Public Derogatory Records", min_value=0, max_value=5, value=0)
        age = st.slider("Applicant Age", min_value=18, max_value=80, value=34)

    st.markdown("---")
    
    # Prepare Single Applicant Data
    applicant_dict = {
        'age': age,
        'annual_income': annual_income,
        'emp_length_years': emp_length_years,
        'home_ownership': home_ownership,
        'loan_amount': loan_amount,
        'loan_intent': loan_intent,
        'loan_term_months': loan_term_months,
        'interest_rate': interest_rate,
        'credit_score': credit_score,
        'dti_ratio': dti_ratio,
        'revolving_utilization': revolving_utilization,
        'total_open_acc': total_open_acc,
        'delinq_2yrs': delinq_2yrs,
        'derogatory_recs': derogatory_recs
    }
    
    applicant_df = pd.DataFrame([applicant_dict])
    
    # Transform and Predict using JSON loaded model
    transformed_applicant = preprocessor.transform(applicant_df)
    prob_default = float(champion_model.predict_proba(transformed_applicant)[0, 1])
    
    # Risk Tier & Decision Logic
    if prob_default < 0.12:
        risk_tier = "Low Risk"
        decision = "AUTO-APPROVE"
        badge_class = "badge-approve"
        tier_color = "#34d399"
    elif prob_default < 0.30:
        risk_tier = "Moderate Risk"
        decision = "MANUAL UNDERWRITING REVIEW"
        badge_class = "badge-review"
        tier_color = "#fbbf24"
    else:
        risk_tier = "High Default Risk"
        decision = "DECLINE / REJECT"
        badge_class = "badge-decline"
        tier_color = "#f87171"
        
    risk_score_100 = int((1.0 - prob_default) * 100)

    # Display Assessment Results Card
    st.markdown("### 📋 Underwriting Risk Assessment")
    
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    with res_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Probability of Default</div>
            <div class="metric-val" style="color: {tier_color};">{prob_default:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">LoanGuard Risk Rating</div>
            <div class="metric-val" style="color: #60a5fa;">{risk_score_100} / 100</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Risk Tier Category</div>
            <div class="metric-val" style="color: {tier_color}; font-size: 1.5rem;">{risk_tier}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Underwriting Recommendation</div>
            <div style="margin-top: 14px;"><span class="{badge_class}">{decision}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SHAP Explainability Section
    st.markdown("### 🔍 SHAP Model Explanation (Key Risk Drivers)")
    
    exp_dict = shap_explainer.explain_single_applicant(transformed_applicant, applicant_dict)
    exp_df = exp_dict['explanation_df'].head(10)
    
    colors = ['#ef4444' if val > 0 else '#10b981' for val in exp_df['shap_value']]
    
    fig_shap = go.Figure(go.Bar(
        x=exp_df['shap_value'],
        y=exp_df['feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in exp_df['shap_value']],
        textposition='outside'
    ))
    
    fig_shap.update_layout(
        title="<b>Top 10 Feature Contributions to Default Risk</b> (Red = Increases Risk, Green = Lowers Risk)",
        xaxis_title="SHAP Value (Impact on Log-Odds of Default)",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig_shap, use_container_width=True)
    
    # Financial Recommendations Box
    if prob_default >= 0.12:
        st.warning("⚠️ **Underwriter Advisory Notes & Risk Mitigation:**")
        drivers = exp_dict['top_risk_drivers']
        for d in drivers[:3]:
            feat_name = d['feature'].replace('num__', '').replace('cat__', '')
            st.markdown(f"- **High Risk Driver ({feat_name})**: Contributes **+{d['shap_value']:.3f}** to default log-odds. Consider requiring additional collateral or reducing loan term.")


# ==========================================
# TAB 2: BATCH PORTFOLIO ANALYTICS
# ==========================================
with tab2:
    st.markdown("### 📊 Portfolio-Level Risk Distribution")
    
    uploaded_file = st.file_uploader("Upload Batch Applicant CSV (or analyze held-out test sample)", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    else:
        batch_df = test_df.copy()
        st.info("💡 Displaying held-out test dataset sample (2,000 applicants).")
        
    X_batch = batch_df.drop(columns=['applicant_id', 'target_default'], errors='ignore')
    X_batch_trans = preprocessor.transform(X_batch)
    
    batch_probs = champion_model.predict_proba(X_batch_trans)[:, 1]
    batch_df['probability_of_default'] = batch_probs
    batch_df['risk_score'] = ((1.0 - batch_probs) * 100).astype(int)
    batch_df['decision'] = np.where(batch_probs < 0.12, 'AUTO-APPROVE', np.where(batch_probs < 0.30, 'MANUAL REVIEW', 'DECLINE'))

    # Summary Metrics
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.metric("Total Applicants", f"{len(batch_df):,}")
    with b_col2:
        st.metric("Average Default Prob.", f"{batch_probs.mean():.2%}")
    with b_col3:
        st.metric("High Risk / Declines", f"{(batch_df['decision'] == 'DECLINE').sum():,} ({(batch_df['decision'] == 'DECLINE').mean():.1%})")
    with b_col4:
        st.metric("Total Loan Capital Requested", f"${batch_df['loan_amount'].sum():,}")

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_hist = px.histogram(
            batch_df, 
            x="probability_of_default", 
            color="decision",
            nbins=30,
            title="<b>Default Probability Distribution</b>",
            color_discrete_map={'AUTO-APPROVE': '#10b981', 'MANUAL REVIEW': '#f59e0b', 'DECLINE': '#ef4444'},
            template="plotly_dark"
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_chart2:
        fig_scatter = px.scatter(
            batch_df,
            x="credit_score",
            y="probability_of_default",
            color="decision",
            size="loan_amount",
            hover_data=["applicant_id", "annual_income", "dti_ratio"],
            title="<b>Credit Score vs. Default Risk</b>",
            color_discrete_map={'AUTO-APPROVE': '#10b981', 'MANUAL REVIEW': '#f59e0b', 'DECLINE': '#ef4444'},
            template="plotly_dark"
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Download Scored Batch
    st.markdown("### 📥 Batch Scored Applicants Table")
    st.dataframe(batch_df[['applicant_id', 'credit_score', 'annual_income', 'loan_amount', 'probability_of_default', 'risk_score', 'decision']].head(100), use_container_width=True)
    
    csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Full Scored Risk CSV",
        data=csv_bytes,
        file_name="loanguard_scored_applicants.csv",
        mime="text/csv"
    )


# ==========================================
# TAB 3: MODEL GOVERNANCE & BENCHMARK
# ==========================================
with tab3:
    st.markdown("### ⚙️ Model Comparison & Governance Benchmarks")
    st.markdown("Evaluation metrics computed on held-out test dataset across 3 classification algorithms:")
    
    # Metrics Cards Table
    m_cols = st.columns(3)
    for idx, (m_name, m_val) in enumerate(metrics.items()):
        with m_cols[idx]:
            is_champion = (m_name == "Logistic Regression")
            badge = " ★ CHAMPION" if is_champion else ""
            st.markdown(f"#### {m_name}{badge}")
            st.markdown(f"""
            - **ROC-AUC Score**: `{m_val['roc_auc']:.4f}`
            - **PR-AUC Score**: `{m_val['pr_auc']:.4f}`
            - **F1 Score**: `{m_val['f1_score']:.4f}`
            - **Precision**: `{m_val['precision']:.4f}`
            - **Recall**: `{m_val['recall']:.4f}`
            - **Optimal Threshold**: `{m_val['optimal_threshold']:.2f}`
            """)

    st.markdown("---")
    
    # Global Feature Importance Plot
    st.markdown("### 🌐 Global SHAP Feature Importance")
    global_importance_df = shap_explainer.get_global_importance(preprocessor.transform(test_df.drop(columns=['applicant_id', 'target_default'])))
    
    fig_global = px.bar(
        global_importance_df.head(12),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="<b>Global Feature Importance</b>",
        labels={"mean_abs_shap": "Mean |SHAP Value| / Impact", "feature": "Feature"},
        template="plotly_dark",
        color="mean_abs_shap",
        color_continuous_scale="Viridis"
    )
    fig_global.update_layout(
        yaxis=dict(autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig_global, use_container_width=True)
