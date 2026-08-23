import json
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List


class CreditRiskExplainer:
    """
    SHAP explainability engine for global model analysis and individual 
    applicant credit decision breakdown with pure JSON serialization support.
    """
    def __init__(self, model: Any, feature_names: List[str], background_data: pd.DataFrame = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        if hasattr(model, 'coef_'):
            coefs = np.ravel(model.coef_)
            intercept = float(np.ravel(model.intercept_)[0])
            self.coef_dict = dict(zip(self.feature_names, [float(c) for c in coefs]))
            self.base_val = intercept
        else:
            self.coef_dict = {}
            self.base_val = 0.0

        # Guarantee non-None background dataset for SHAP maskers
        if background_data is None:
            background_data = pd.DataFrame(np.zeros((10, len(feature_names))), columns=feature_names)

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            try:
                sample_bg = background_data.sample(min(50, len(background_data)), random_state=42)
                self.explainer = shap.Explainer(self.model.predict_proba, masker=sample_bg)
            except Exception:
                self.explainer = None

    def get_global_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean absolute SHAP values across a dataset sample.
        """
        if self.coef_dict and self.explainer is None:
            vals = [abs(self.coef_dict.get(f, 0.0)) for f in self.feature_names]
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': vals
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
            return importance_df

        if self.explainer is None:
            vals = [0.1] * len(self.feature_names)
            return pd.DataFrame({'feature': self.feature_names, 'mean_abs_shap': vals})

        try:
            shap_values = self.explainer(X_sample)
            vals_matrix = shap_values.values
        except Exception:
            shap_values = self.explainer.shap_values(X_sample)
            vals_matrix = shap_values

        if isinstance(vals_matrix, list):
            vals_matrix = vals_matrix[1] if len(vals_matrix) > 1 else vals_matrix[0]
            
        if len(vals_matrix.shape) == 3:
            vals = np.abs(vals_matrix[:, :, 1]).mean(axis=0)
        elif len(vals_matrix.shape) == 2:
            vals = np.abs(vals_matrix).mean(axis=0)
        else:
            vals = np.abs(vals_matrix).mean(axis=0)
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': vals
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        return importance_df

    def explain_single_applicant(
        self, applicant_row_transformed: pd.DataFrame, applicant_raw_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates individual SHAP waterfall values for a single loan applicant.
        """
        if self.explainer is not None:
            try:
                shap_exp = self.explainer(applicant_row_transformed)
                if hasattr(shap_exp, 'values'):
                    vals_matrix = shap_exp.values
                    base_val = shap_exp.base_values
                else:
                    vals_matrix = shap_exp
                    base_val = 0.0
            except Exception:
                try:
                    vals_matrix = self.explainer.shap_values(applicant_row_transformed)
                    base_val = getattr(self.explainer, 'expected_value', 0.0)
                except Exception:
                    vals_matrix = None
                    base_val = 0.0

            if vals_matrix is not None:
                if isinstance(vals_matrix, list):
                    vals_matrix = vals_matrix[1] if len(vals_matrix) > 1 else vals_matrix[0]
                    if isinstance(base_val, list):
                        base_val = base_val[1] if len(base_val) > 1 else base_val[0]

                if len(vals_matrix.shape) == 3:
                    vals = vals_matrix[0, :, 1]
                    base_val = base_val[0, 1] if hasattr(base_val, 'shape') and len(base_val.shape) > 1 else base_val
                elif len(vals_matrix.shape) == 2:
                    vals = vals_matrix[0]
                    base_val = base_val[0] if isinstance(base_val, (list, np.ndarray)) else base_val
                else:
                    vals = vals_matrix

                base_val_scalar = float(np.ravel(base_val)[0]) if hasattr(base_val, '__iter__') else float(base_val)
            else:
                vals = [self.coef_dict.get(f, 0.0) * float(applicant_row_transformed[f].values[0]) for f in self.feature_names]
                base_val_scalar = self.base_val
        else:
            vals = [self.coef_dict.get(f, 0.0) * float(applicant_row_transformed[f].values[0]) for f in self.feature_names]
            base_val_scalar = self.base_val

        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': vals,
            'feature_value': applicant_row_transformed.iloc[0].values
        })
        
        explanation_df['abs_shap'] = explanation_df['shap_value'].abs()
        explanation_df = explanation_df.sort_values('abs_shap', ascending=False).reset_index(drop=True)
        
        top_risk_drivers = explanation_df[explanation_df['shap_value'] > 0].head(5).to_dict('records')
        top_protective_factors = explanation_df[explanation_df['shap_value'] < 0].head(5).to_dict('records')
        
        return {
            'base_value': base_val_scalar,
            'explanation_df': explanation_df,
            'top_risk_drivers': top_risk_drivers,
            'top_protective_factors': top_protective_factors
        }

    def to_json(self, filepath: str):
        """
        Exports explainer configuration to pure JSON file.
        """
        data = {
            'feature_names': self.feature_names,
            'coef_dict': self.coef_dict,
            'base_val': self.base_val
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Exported SHAP explainer to JSON: {filepath}")

    @classmethod
    def from_json(cls, model: Any, feature_names: List[str], filepath: str):
        """
        Loads explainer configuration from pure JSON file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        explainer = cls(model=model, feature_names=data['feature_names'])
        explainer.coef_dict = data.get('coef_dict', {})
        explainer.base_val = data.get('base_val', 0.0)
        return explainer
