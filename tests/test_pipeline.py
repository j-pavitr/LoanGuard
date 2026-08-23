import os
import unittest
import numpy as np
import pandas as pd

from src.data_generator import generate_credit_dataset
from src.data_processing import CreditDataPreprocessor, FeatureEngineer
from src.model_trainer import CreditRiskModelTrainer, save_model_to_json, load_model_from_json
from src.explainability import CreditRiskExplainer


class TestLoanGuardPipeline(unittest.TestCase):

    def setUp(self):
        self.df = generate_credit_dataset(num_samples=200, random_state=42)
        self.X = self.df.drop(columns=['applicant_id', 'target_default'])
        self.y = self.df['target_default']
        os.makedirs("scratch_test", exist_ok=True)

    def test_data_generation(self):
        self.assertEqual(len(self.df), 200)
        self.assertIn('target_default', self.df.columns)
        self.assertIn('annual_income', self.df.columns)
        self.assertIn('credit_score', self.df.columns)

    def test_feature_engineer(self):
        fe = FeatureEngineer()
        transformed = fe.transform(self.X)
        self.assertIn('loan_to_income_ratio', transformed.columns)
        self.assertIn('estimated_monthly_payment', transformed.columns)
        self.assertIn('installment_to_income_ratio', transformed.columns)
        self.assertIn('delinquency_severity_score', transformed.columns)
        self.assertIn('credit_score_tier', transformed.columns)

    def test_preprocessor_json(self):
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.X)
        self.assertEqual(len(X_trans), 200)
        
        json_path = "scratch_test/preprocessor.json"
        preprocessor.to_json(json_path)
        
        loaded_prep = CreditDataPreprocessor.from_json(json_path)
        X_trans_loaded = loaded_prep.transform(self.X)
        self.assertEqual(X_trans.shape, X_trans_loaded.shape)

    def test_model_trainer_json(self):
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.X)
        
        trainer = CreditRiskModelTrainer(random_state=42)
        metrics_dict, trained_models = trainer.train_and_compare(
            X_trans, self.y, X_trans, self.y, use_smote=False
        )
        
        best_model = trainer.best_model
        model_json_path = "scratch_test/champion_model.json"
        save_model_to_json(best_model, model_json_path, feature_names=list(X_trans.columns))
        
        loaded_model = load_model_from_json(model_json_path)
        probs = loaded_model.predict_proba(X_trans)
        self.assertEqual(len(probs), 200)

    def test_explainability_json(self):
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.X)
        
        trainer = CreditRiskModelTrainer(random_state=42)
        _, trained_models = trainer.train_and_compare(X_trans, self.y, X_trans, self.y, use_smote=False)
        best_model = trainer.best_model
        
        explainer = CreditRiskExplainer(best_model, list(X_trans.columns))
        explainer_json_path = "scratch_test/shap_explainer.json"
        explainer.to_json(explainer_json_path)
        
        loaded_explainer = CreditRiskExplainer.from_json(best_model, list(X_trans.columns), explainer_json_path)
        single_exp = loaded_explainer.explain_single_applicant(X_trans.head(1), self.X.iloc[0].to_dict())
        self.assertIn('top_risk_drivers', single_exp)
        self.assertIn('top_protective_factors', single_exp)


if __name__ == '__main__':
    unittest.main()
