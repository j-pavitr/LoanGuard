import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def calculate_monthly_payment(principal, annual_interest_rate_pct, term_months):
    """
    Calculates monthly payment using standard amortization formula.
    """
    rate = (annual_interest_rate_pct / 100.0) / 12.0
    rate = np.maximum(rate, 1e-6)
    payment = principal * (rate * (1 + rate)**term_months) / ((1 + rate)**term_months - 1)
    return payment


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Engineers derived credit risk features:
    1. loan_to_income_ratio
    2. estimated_monthly_payment
    3. installment_to_income_ratio
    4. delinquency_severity_score
    5. high_utilization_flag
    6. credit_score_tier
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        income = df['annual_income'].fillna(df['annual_income'].median() if 'annual_income' in df else 50000)
        income_monthly = np.maximum(income / 12.0, 100.0)
        
        df['loan_to_income_ratio'] = (df['loan_amount'] / np.maximum(income, 1.0)).round(4)
        
        df['estimated_monthly_payment'] = calculate_monthly_payment(
            df['loan_amount'], df['interest_rate'], df['loan_term_months']
        ).round(2)
        
        df['installment_to_income_ratio'] = (
            df['estimated_monthly_payment'] / income_monthly
        ).round(4)
        
        df['delinquency_severity_score'] = (
            df['delinq_2yrs'] * 1.5 + df['derogatory_recs'] * 2.5
        ).round(2)
        
        df['high_utilization_flag'] = (df['revolving_utilization'] > 70.0).astype(int)
        
        bins = [0, 580, 660, 740, 850]
        labels = ['POOR', 'FAIR', 'GOOD', 'EXCELLENT']
        df['credit_score_tier'] = pd.cut(
            df['credit_score'], bins=bins, labels=labels, include_lowest=True
        ).astype(str)
        
        return df


class CreditDataPreprocessor:
    """
    Full Preprocessing pipeline that performs imputation, feature engineering, 
    scaling, encoding, and supports pure JSON serialization.
    """
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.numeric_features = [
            'age', 'annual_income', 'emp_length_years', 'loan_amount',
            'loan_term_months', 'interest_rate', 'credit_score',
            'dti_ratio', 'revolving_utilization', 'total_open_acc',
            'delinq_2yrs', 'derogatory_recs', 'loan_to_income_ratio',
            'estimated_monthly_payment', 'installment_to_income_ratio',
            'delinquency_severity_score', 'high_utilization_flag'
        ]
        self.categorical_features = ['home_ownership', 'loan_intent', 'credit_score_tier']
        
        self.preprocessor = None
        self.feature_names = []
        
        # Extracted parameters for JSON persistence
        self.num_medians = {}
        self.num_means = {}
        self.num_scales = {}
        self.cat_modes = {}
        self.cat_categories = {}

    def _build_pipeline(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

    def fit(self, X, y=None):
        X_engineered = self.feature_engineer.transform(X)
        self._build_pipeline()
        self.preprocessor.fit(X_engineered)
        
        # Extract parameters for JSON export
        num_imputer = self.preprocessor.named_transformers_['num'].named_steps['imputer']
        num_scaler = self.preprocessor.named_transformers_['num'].named_steps['scaler']
        cat_imputer = self.preprocessor.named_transformers_['cat'].named_steps['imputer']
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
        
        self.num_medians = dict(zip(self.numeric_features, [float(v) for v in num_imputer.statistics_]))
        self.num_means = dict(zip(self.numeric_features, [float(v) for v in num_scaler.mean_]))
        self.num_scales = dict(zip(self.numeric_features, [float(v) for v in num_scaler.scale_]))
        self.cat_modes = dict(zip(self.categorical_features, [str(v) for v in cat_imputer.statistics_]))
        
        self.cat_categories = {}
        for col_name, cats in zip(self.categorical_features, cat_encoder.categories_):
            self.cat_categories[col_name] = [str(c) for c in cats]
            
        encoded_cat_names = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        self.feature_names = self.numeric_features + encoded_cat_names
        return self

    def transform(self, X):
        X_engineered = self.feature_engineer.transform(X)
        
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(X_engineered)
            return pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        
        # Custom JSON transform implementation
        transformed_rows = []
        for idx, row in X_engineered.iterrows():
            row_dict = {}
            # Process numeric features
            for col in self.numeric_features:
                val = row[col] if col in row and not pd.isna(row[col]) else self.num_medians.get(col, 0.0)
                mean_val = self.num_means.get(col, 0.0)
                scale_val = self.num_scales.get(col, 1.0)
                if scale_val == 0:
                    scale_val = 1.0
                row_dict[col] = (float(val) - mean_val) / scale_val
                
            # Process categorical features
            for col in self.categorical_features:
                raw_val = str(row[col]) if col in row and not pd.isna(row[col]) else self.cat_modes.get(col, '')
                cats = self.cat_categories.get(col, [])
                for cat in cats:
                    dummy_name = f"{col}_{cat}"
                    row_dict[dummy_name] = 1.0 if raw_val == cat else 0.0
                    
            transformed_rows.append(row_dict)
            
        res_df = pd.DataFrame(transformed_rows, index=X.index)
        # Ensure exact column ordering
        res_df = res_df.reindex(columns=self.feature_names, fill_value=0.0)
        return res_df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def to_json(self, filepath: str):
        """
        Exports preprocessor state to pure JSON file.
        """
        data = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'num_medians': self.num_medians,
            'num_means': self.num_means,
            'num_scales': self.num_scales,
            'cat_modes': self.cat_modes,
            'cat_categories': self.cat_categories
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Exported preprocessor to JSON: {filepath}")

    @classmethod
    def from_json(cls, filepath: str):
        """
        Loads preprocessor state from pure JSON file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        prep = cls()
        prep.numeric_features = data['numeric_features']
        prep.categorical_features = data['categorical_features']
        prep.feature_names = data['feature_names']
        prep.num_medians = data['num_medians']
        prep.num_means = data['num_means']
        prep.num_scales = data['num_scales']
        prep.cat_modes = data['cat_modes']
        prep.cat_categories = data['cat_categories']
        return prep
