import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


class JSONLogisticRegressionWrapper:
    """
    Pure JSON serializable Logistic Regression model.
    """
    def __init__(self, coef, intercept, classes, feature_names=None):
        self.coef_ = np.array(coef)
        self.intercept_ = np.array(intercept)
        self.classes_ = np.array(classes)
        self.feature_names = feature_names

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        z = np.dot(X_arr, self.coef_.T) + self.intercept_
        prob_1 = 1.0 / (1.0 + np.exp(-z))
        prob_1 = prob_1.ravel()
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class JSONRandomForestWrapper:
    """
    Pure JSON serializable Random Forest model wrapper.
    """
    def __init__(self, feature_importances, classes, feature_names=None):
        self.feature_importances_ = np.array(feature_importances)
        self.classes_ = np.array(classes)
        self.feature_names = feature_names

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        # Linear approximation from feature importances for predictions
        z = np.dot(X_arr, self.feature_importances_)
        prob_1 = 1.0 / (1.0 + np.exp(-z))
        prob_1 = np.clip(prob_1, 0.01, 0.99)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def save_model_to_json(model: Any, filepath: str, feature_names: list = None):
    """
    Saves a trained classification model to pure JSON format.
    """
    if isinstance(model, LogisticRegression):
        data = {
            'model_type': 'LogisticRegression',
            'coef': model.coef_.tolist(),
            'intercept': model.intercept_.tolist(),
            'classes': model.classes_.tolist(),
            'feature_names': feature_names
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    elif isinstance(model, XGBClassifier):
        model.save_model(filepath)
        
    elif isinstance(model, RandomForestClassifier):
        data = {
            'model_type': 'RandomForestClassifier',
            'feature_importances': model.feature_importances_.tolist(),
            'classes': model.classes_.tolist(),
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'feature_names': feature_names
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    elif isinstance(model, JSONLogisticRegressionWrapper):
        data = {
            'model_type': 'LogisticRegression',
            'coef': model.coef_.tolist(),
            'intercept': model.intercept_.tolist(),
            'classes': model.classes_.tolist(),
            'feature_names': model.feature_names
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    else:
        raise ValueError(f"Unsupported model type for JSON export: {type(model)}")
        
    print(f"Exported model to JSON: {filepath}")


def load_model_from_json(filepath: str) -> Any:
    """
    Loads a model from a JSON file.
    """
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            data = None
            
    if data and isinstance(data, dict):
        m_type = data.get('model_type')
        if m_type == 'LogisticRegression':
            return JSONLogisticRegressionWrapper(
                coef=data['coef'],
                intercept=data['intercept'],
                classes=data['classes'],
                feature_names=data.get('feature_names')
            )
        elif m_type == 'RandomForestClassifier':
            return JSONRandomForestWrapper(
                feature_importances=data['feature_importances'],
                classes=data['classes'],
                feature_names=data.get('feature_names')
            )
            
    # Try loading native XGBoost JSON
    xgb = XGBClassifier()
    xgb.load_model(filepath)
    return xgb


class CreditRiskModelTrainer:
    """
    Trains, tunes, handles class imbalance, and evaluates classification models 
    for credit default risk prediction.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model_name = None
        self.best_model = None

    def get_model_instances(self, scale_pos_weight: float = 10.0) -> Dict[str, Any]:
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state,
                C=0.5
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            )
        }

    def evaluate_model(
        self, model: Any, X_val: pd.DataFrame, y_val: pd.Series, threshold: float = 0.50
    ) -> Dict[str, Any]:
        y_probs = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_probs)
        precision_vec, recall_vec, thresholds_vec = precision_recall_curve(y_val, y_probs)
        pr_auc = auc(recall_vec, precision_vec)
        
        f1_scores = 2 * (precision_vec * recall_vec) / np.maximum(precision_vec + recall_vec, 1e-6)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds_vec[best_idx]) if best_idx < len(thresholds_vec) else 0.50
        
        y_preds = (y_probs >= threshold).astype(int)
        
        f1 = f1_score(y_val, y_preds)
        prec = precision_score(y_val, y_preds, zero_division=0)
        rec = recall_score(y_val, y_preds, zero_division=0)
        cm = confusion_matrix(y_val, y_preds).tolist()
        
        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1_score': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'optimal_threshold': float(optimal_threshold),
            'best_f1_at_optimal_threshold': float(f1_scores[best_idx]),
            'confusion_matrix': cm,
            'y_probs': y_probs
        }

    def train_and_compare(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, use_smote: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        neg_pos_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        raw_models = self.get_model_instances(scale_pos_weight=neg_pos_ratio)
        
        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
            
        evaluation_results = {}
        best_score = -1.0
        
        for name, model in raw_models.items():
            model.fit(X_train_res, y_train_res)
            metrics = self.evaluate_model(model, X_val, y_val)
            
            self.models[name] = model
            metrics_to_save = {k: v for k, v in metrics.items() if k != 'y_probs'}
            evaluation_results[name] = metrics_to_save
            
            composite_score = 0.60 * metrics['roc_auc'] + 0.40 * metrics['pr_auc']
            
            if composite_score > best_score:
                best_score = composite_score
                self.best_model_name = name
                self.best_model = model
                
        return evaluation_results, self.models
