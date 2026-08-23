import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_generator import save_dataset
from src.data_processing import CreditDataPreprocessor
from src.model_trainer import CreditRiskModelTrainer, save_model_to_json
from src.explainability import CreditRiskExplainer


def run_pipeline():
    print("=" * 60)
    print("      LOANGUARD - CREDIT RISK MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Dataset Generation & Loading
    data_dir = "data"
    models_dir = "models"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    csv_path = save_dataset(output_dir=data_dir)
    df = pd.read_csv(csv_path)
    print(f"\n[1/5] Loaded dataset with shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=['applicant_id', 'target_default'])
    y = df['target_default']
    
    # 2. Train / Test Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2/5] Stratified Train/Test split complete.")
    print(f"      Train set: {X_train.shape[0]} samples (Defaults: {y_train.sum()} / {y_train.mean():.2%})")
    print(f"      Test set:  {X_test.shape[0]} samples (Defaults: {y_test.sum()} / {y_test.mean():.2%})")
    
    # 3. Fit Preprocessing Pipeline & Engineering Features
    print("\n[3/5] Fitting Preprocessor & Feature Engineering...")
    preprocessor = CreditDataPreprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"      Engineered & Transformed Feature shape: {X_train_transformed.shape}")
    print(f"      Features: {list(X_train_transformed.columns)}")
    
    # Save preprocessor as JSON
    preprocessor.to_json(os.path.join(models_dir, "preprocessor.json"))
    
    # 4. Model Training & Imbalance Handling Comparison
    print("\n[4/5] Training & Evaluating Classification Models (Logistic Regression, Random Forest, XGBoost)...")
    trainer = CreditRiskModelTrainer(random_state=42)
    evaluation_results, trained_models = trainer.train_and_compare(
        X_train_transformed, y_train, X_test_transformed, y_test, use_smote=True
    )
    
    print("\n" + "-" * 60)
    print("                 MODEL EVALUATION SUMMARY")
    print("-" * 60)
    for model_name, metrics in evaluation_results.items():
        print(f"-> Model: {model_name:20s}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | Opt. Threshold: {metrics['optimal_threshold']:.2f}")
        print("-" * 60)
        
    best_model_name = trainer.best_model_name
    best_model = trainer.best_model
    print(f"\n[+] Champion Model Selected: {best_model_name}")
    
    # Save models & comparison metrics as JSON
    save_model_to_json(
        best_model, 
        os.path.join(models_dir, "champion_model.json"), 
        feature_names=list(X_train_transformed.columns)
    )
    
    # Save evaluation metrics JSON
    with open(os.path.join(models_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)
        
    # Save test sample data for UI benchmarking
    test_sample = X_test.copy()
    test_sample['target_default'] = y_test
    test_sample['applicant_id'] = df.loc[X_test.index, 'applicant_id']
    test_sample.to_csv(os.path.join(data_dir, "test_sample.csv"), index=False)
    
    # 5. SHAP Explainer Setup & Export to JSON
    print("\n[5/5] Initializing SHAP Explainer engine...")
    explainer = CreditRiskExplainer(
        best_model, list(X_train_transformed.columns), background_data=X_train_transformed
    )
    explainer.to_json(os.path.join(models_dir, "shap_explainer.json"))
    
    print("\n" + "=" * 60)
    print("   SUCCESS: LOANGUARD PIPELINE EXECUTION COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
