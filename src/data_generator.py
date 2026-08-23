import os
import numpy as np
import pandas as pd


def generate_credit_dataset(num_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit default risk dataset based on standard 
    LendingClub and Kaggle 'Give Me Some Credit' tabular data distribution patterns.
    """
    np.random.seed(random_state)
    
    applicant_ids = [f"APP-{10000 + i}" for i in range(num_samples)]
    
    # Demographic & Income attributes
    age = np.random.randint(21, 70, size=num_samples)
    annual_income = np.random.lognormal(mean=11.0, sigma=0.5, size=num_samples)
    annual_income = np.clip(annual_income, 18000, 350000).round(2)
    
    emp_length_years = np.random.choice(
        np.arange(0, 21), size=num_samples, 
        p=[0.12] + [0.044]*20
    )
    
    home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN', 'OTHER'], 
        size=num_samples, 
        p=[0.45, 0.43, 0.10, 0.02]
    )
    
    # Loan parameters
    loan_amount = np.random.choice(
        np.arange(2000, 40001, 1000), 
        size=num_samples
    )
    
    loan_intent = np.random.choice(
        ['DEBTCONSOLIDATION', 'PERSONAL', 'HOMEIMPROVEMENT', 'VENTURE', 'MEDICAL', 'EDUCATION'],
        size=num_samples,
        p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]
    )
    
    loan_term_months = np.random.choice([36, 60], size=num_samples, p=[0.70, 0.30])
    
    # Base credit score (300 to 850)
    credit_score = np.random.normal(loc=690, scale=60, size=num_samples).astype(int)
    credit_score = np.clip(credit_score, 350, 850)
    
    # Interest rate depends inversely on credit score + noise
    base_ir = 25.0 - (credit_score - 350) * (18.0 / 500)
    interest_rate = base_ir + np.random.normal(0, 1.5, size=num_samples)
    interest_rate = np.clip(interest_rate, 5.0, 28.0).round(2)
    
    # Financial metrics
    dti_ratio = np.random.gamma(shape=3.0, scale=6.0, size=num_samples).round(2)
    dti_ratio = np.clip(dti_ratio, 2.0, 65.0)
    
    revolving_utilization = np.random.beta(a=2, b=3, size=num_samples) * 100
    revolving_utilization = revolving_utilization.round(2)
    
    total_open_acc = np.random.poisson(lam=10, size=num_samples)
    total_open_acc = np.clip(total_open_acc, 1, 40)
    
    # Delinquencies and Derogatory Records (rare, zero-inflated)
    delinq_2yrs = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.82, 0.10, 0.04, 0.02, 0.01, 0.01])
    derogatory_recs = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.90, 0.07, 0.02, 0.01])
    
    # Generate realistic Logit for Default Risk (Probability of Default)
    # High risk drivers: low credit score, high DTI, high utilization, past delinquencies, low income vs loan amount
    loan_to_income = loan_amount / annual_income
    
    logit = (
        -3.5
        - 0.012 * (credit_score - 650)
        + 0.045 * (dti_ratio - 18)
        + 0.030 * (revolving_utilization - 40)
        + 0.45 * delinq_2yrs
        + 0.60 * derogatory_recs
        + 1.80 * (loan_to_income - 0.25)
        + 0.08 * (interest_rate - 12)
        + np.random.normal(0, 0.6, size=num_samples)
    )
    
    prob_default = 1 / (1 + np.exp(-logit))
    
    # Target variable: binary default flag (~8.5% default rate)
    target_default = (np.random.uniform(0, 1, size=num_samples) < prob_default).astype(int)
    
    # Introduce small realistic missing values (~1-2% missing in income and emp_length)
    mask_income = np.random.rand(num_samples) < 0.015
    annual_income_with_nan = annual_income.copy()
    annual_income_with_nan[mask_income] = np.nan
    
    df = pd.DataFrame({
        'applicant_id': applicant_ids,
        'age': age,
        'annual_income': annual_income_with_nan,
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
        'derogatory_recs': derogatory_recs,
        'target_default': target_default
    })
    
    return df


def save_dataset(output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = generate_credit_dataset(num_samples=10000, random_state=42)
    file_path = os.path.join(output_dir, "loan_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Generated {len(df)} records. Saved to {file_path}")
    print(f"Default rate: {df['target_default'].mean():.2%}")
    return file_path


if __name__ == "__main__":
    save_dataset()
