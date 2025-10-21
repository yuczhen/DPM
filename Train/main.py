# -*- coding: utf-8 -*-
"""
DPM Model Training Pipeline (Clean Version - No W&B)
===============================================
Complete machine learning pipeline including:
- Advanced feature engineering
- WoE encoding
- Geographic risk features
- Overdue pattern analysis (Tier 1 improvements)
- Model ensemble
- SHAP analysis
- Model persistence

For W&B tracking version, use main_wandb.py instead.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning packages
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)

# Advanced models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# SHAP for interpretability
import shap

# Utilities
import joblib
import json
from datetime import datetime
import os
from scipy import stats
import sys
sys.path.append('..')
import config

# Import feature engineering from dedicated module
from feature_engineering import (
    WoEEncoder,
    TargetEncoder,
    GeographicRiskEncoder,
    OverduePatternEncoder,
    get_feature_lists
)

# Matplotlib Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AdvancedDefaultPredictionPipeline:
    """
    Advanced Default Prediction Pipeline for client risk assessment
    Clean version without W&B dependency - suitable for sharing with team
    """

    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.models = {}
        self.feature_engineered_data = None
        self.feature_names = None
        self.target_col = 'Default'
        self.categorical_features = []
        self.numerical_features = []
        self.best_model = None
        self.ensemble_model = None
        self.feature_importance_df = None
        self.shap_explainers = {}

        # Set random seed
        np.random.seed(random_state)

        # WoE encoder
        self.woe_encoder = None

    def load_real_data(self, file_path='source/DPM_merged_cleaned.xlsx'):
        """Load real DPM data"""
        print("=" * 70)
        print("Loading Real Data")
        print("=" * 70)

        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"\nOriginal data: {len(df):,} rows, {len(df.columns)} columns")

        return df

    def handle_missing_values(self, df):
        """Handle missing values"""
        print("\n" + "=" * 70)
        print("Handling Missing Values")
        print("=" * 70)

        df_clean = df.copy()

        # Find column names (handle line breaks)
        def find_col(keyword):
            matches = [col for col in df_clean.columns if keyword in str(col).lower()]
            return matches[0] if matches else None

        # 1. Overdue columns -> fill with 0
        print("\n[1/6] Overdue columns")
        overdue_cols = [
            'paid installments',
            'number of overdue before the first month',
            'number of overdue in the first half of the first month',
            'number of overdue in the second half of the first month',
            'number of overdue in the second month',
            'number of overdue in the third month',
            'number of overdue in the fourth month',
            'number of overdue in the fifth month',
            'number of overdue in the sixth month',
        ]

        for col in overdue_cols:
            if col in df_clean.columns:
                before = df_clean[col].isnull().sum()
                if before > 0:
                    df_clean[col] = df_clean[col].fillna(0)
                    print(f"  {col}: filled {before} missing values (with 0)")

        # 2. Month salary -> fill with median
        print("\n[2/6] Month salary")
        salary_col = find_col('salary')
        if salary_col:
            df_clean[salary_col] = pd.to_numeric(df_clean[salary_col], errors='coerce')
            salary_median = df_clean[df_clean.index != 0][salary_col].median()
            before = df_clean[salary_col].isnull().sum()
            if before > 0:
                df_clean[salary_col] = df_clean[salary_col].fillna(salary_median)
                print(f"  month salary: filled {before} missing values (median: {salary_median:.2f})")

        # 3. Education -> fill with mode
        print("\n[3/6] Education")
        edu_col = find_col('education')
        if edu_col:
            edu_mode = df_clean[edu_col].mode()[0] if len(df_clean[edu_col].mode()) > 0 else '高中/職'
            before = df_clean[edu_col].isnull().sum()
            if before > 0:
                df_clean[edu_col] = df_clean[edu_col].fillna(edu_mode)
                print(f"  education: filled {before} missing values (mode: {edu_mode})")

        # 4. Residence status -> fill with mode
        print("\n[4/6] Residence status")
        residence_col = find_col('residence status')
        if residence_col:
            residence_mode = df_clean[residence_col].mode()[0] if len(df_clean[residence_col].mode()) > 0 else '租賃'
            before = df_clean[residence_col].isnull().sum()
            if before > 0:
                df_clean[residence_col] = df_clean[residence_col].fillna(residence_mode)
                print(f"  residence status: filled {before} missing values (mode: {residence_mode})")

        # 5. Main business -> fill with '其他'
        print("\n[5/6] Main business")
        business_col = find_col('main business')
        if business_col:
            before = df_clean[business_col].isnull().sum()
            if before > 0:
                df_clean[business_col] = df_clean[business_col].fillna('其他')
                print(f"  main business: filled {before} missing values (with '其他')")

        # 6. Other columns
        print("\n[6/6] Other columns")

        # Job tenure -> median
        tenure_col = find_col('job tenure')
        if tenure_col:
            df_clean[tenure_col] = pd.to_numeric(df_clean[tenure_col], errors='coerce')
            tenure_median = df_clean[tenure_col].median()
            before = df_clean[tenure_col].isnull().sum()
            if before > 0:
                df_clean[tenure_col] = df_clean[tenure_col].fillna(tenure_median)
                print(f"  job tenure: filled {before} missing values (median: {tenure_median:.1f})")

        # Marriage -> mode
        marriage_col = find_col('marriage')
        if marriage_col:
            marriage_mode = df_clean[marriage_col].mode()[0] if len(df_clean[marriage_col].mode()) > 0 else '未婚'
            before = df_clean[marriage_col].isnull().sum()
            if before > 0:
                df_clean[marriage_col] = df_clean[marriage_col].fillna(marriage_mode)
                print(f"  marriage: filled {before} missing values (mode: {marriage_mode})")

        print("\nMissing value handling complete!")
        return df_clean

    def convert_data_types(self, df):
        """Convert data types"""
        print("\n" + "=" * 70)
        print("Data Type Conversion")
        print("=" * 70)

        df_typed = df.copy()

        # Numerical columns
        numeric_cols = [
            'loan term', 'paid installments', 'month salary', 'job tenure',
            'overdue days',
            'number of overdue before the first month',
            'number of overdue in the first half of the first month',
            'number of overdue in the second half of the first month',
            'number of overdue in the second month',
            'number of overdue in the third month',
            'number of overdue in the fourth month',
            'number of overdue in the fifth month',
            'number of overdue in the sixth month',
        ]

        print("\nNumerical columns:")
        for col in numeric_cols:
            if col in df_typed.columns:
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
                print(f"  {col}")

        # Categorical columns
        categorical_cols = [
            'overdue status', 'education', 'residence status', 'main business',
            'product', 'marriage', 'post code of residential address',
            'post code of permanent address',
        ]

        print("\nCategorical columns:")
        for col in categorical_cols:
            if col in df_typed.columns:
                df_typed[col] = df_typed[col].astype(str)
                print(f"  {col}")

        return df_typed

    def define_default(self, df):
        """Define default label - M1+ (overdue 30+ days) = default"""
        print("\n" + "=" * 70)
        print("Define Default Label (M1+ Definition)")
        print("=" * 70)

        def classify_default(overdue_status):
            status_str = str(overdue_status).strip()
            # M1+ definition: Current, M0 = Normal (0), M1+ = Default (1)
            if status_str in ['Current', 'M0']:
                return 0
            else:
                try:
                    if status_str.startswith('M'):
                        m_number = int(status_str[1:])
                        return 1 if m_number >= 1 else 0
                except:
                    pass
                return 0

        df['Default'] = df['overdue status'].apply(classify_default)

        print(f"\nTotal samples: {len(df):,}")
        print(f"Default samples: {df['Default'].sum():,} ({df['Default'].mean():.2%})")
        print(f"Normal samples: {(df['Default']==0).sum():,} ({(df['Default']==0).mean():.2%})")

        return df

    def credit_focused_feature_engineering(self, df):
        """
        Feature Engineering - Credit Assessment Focus + DTI + Tier 1 Overdue Patterns

        Focus areas:
        1. Repayment capability (financial)
        2. Contact stability
        3. DTI (debt-to-income) - CRITICAL
        4. Overdue behavior patterns - TIER 1 ENHANCED
        """
        print("\n" + "=" * 70)
        print("Feature Engineering (Credit Assessment Focus + DTI + Tier 1)")
        print("=" * 70)

        df_fe = df.copy()

        # === Financial Capability ===
        print("\n[1/5] Financial Capability Indicators")

        # 1. Payment progress ratio
        df_fe['payment_progress_ratio'] = (
            df_fe['paid installments'] / df_fe['loan term']
        ).fillna(0).clip(0, 1)
        print("  + payment_progress_ratio (paid/total installments)")

        # 2. Job stability (tenure >= 1 year)
        df_fe['job_stable'] = (df_fe['job tenure'] >= 1).astype(int)
        print("  + job_stable (tenure >= 1 year)")

        # === Contact Stability ===
        print("\n[2/5] Contact Stability Indicators")

        # 3. Address match
        df_fe['address_match'] = (
            df_fe['post code of permanent address'] ==
            df_fe['post code of residential address']
        ).astype(int)
        print("  + address_match (permanent = residential)")

        # 4. Residence stability
        stable_residence = ['自有', '配偶名下', '親友名下']
        df_fe['residence_stable'] = (
            df_fe['residence status'].isin(stable_residence)
        ).astype(int)
        print("  + residence_stable (own/spouse/family)")

        # === DTI & Financial Burden - CRITICAL! ===
        print("\n[3/5] DTI & Financial Burden (CRITICAL!)")

        # 5. DTI ratio
        if 'debt_to_income_ratio' in df_fe.columns:
            df_fe['dti_ratio'] = pd.to_numeric(df_fe['debt_to_income_ratio'], errors='coerce').fillna(0).clip(0, 2)
            print("  + dti_ratio (debt/income ratio - PRIMARY INDICATOR)")

        # 6. Payment pressure
        if 'payment_to_income_ratio' in df_fe.columns:
            df_fe['payment_pressure'] = pd.to_numeric(df_fe['payment_to_income_ratio'], errors='coerce').fillna(0).clip(0, 1)
            print("  + payment_pressure (payment/income ratio)")

        # === Geographic Risk Features ===
        print("\n[4/5] Geographic Risk Features")
        geo_encoder = GeographicRiskEncoder(min_samples=10)
        df_fe = geo_encoder.fit_transform(df_fe, df_fe.get('Default', pd.Series([0]*len(df_fe))))

        # === Tier 1: Overdue Pattern Features (ENHANCED!) ===
        print("\n[5/5] Tier 1: Enhanced Overdue Pattern Analysis")
        overdue_encoder = OverduePatternEncoder()
        df_fe = overdue_encoder.create_overdue_pattern_features(df_fe)

        print(f"\nFeature engineering completed!")
        print(f"  Original features: {len(df.columns)}")
        print(f"  Total features after engineering: {len(df_fe.columns)}")
        print(f"  New features added: {len(df_fe.columns) - len(df.columns)}")

        return df_fe

    def prepare_features_for_modeling(self, df):
        """Prepare features for modeling"""
        print("\n" + "=" * 70)
        print("Prepare Features for Modeling")
        print("=" * 70)

        # Get feature lists from shared module
        categorical_features, numerical_features, optional_features = get_feature_lists()

        # Filter features that actually exist in the dataframe
        categorical_features = [f for f in categorical_features if f in df.columns]
        numerical_features = [f for f in numerical_features if f in df.columns]

        print(f"\nCategorical features ({len(categorical_features)}):")
        for feat in categorical_features:
            unique_count = df[feat].nunique()
            print(f"  - {feat:45s} ({unique_count:4d} unique values)")

        print(f"\nNumerical features ({len(numerical_features)}):")
        for feat in numerical_features:
            print(f"  - {feat}")

        return categorical_features, numerical_features

    def initialize_models(self):
        """Initialize models with optimized hyperparameters from config.py"""
        self.models = {
            'XGBoost': xgb.XGBClassifier(**config.MODEL_PARAMS['XGBoost']),
            'LightGBM': lgb.LGBMClassifier(**config.MODEL_PARAMS['LightGBM']),
            'CatBoost': CatBoostClassifier(**config.MODEL_PARAMS['CatBoost'])
        }

        print("\nModels initialized with config.py hyperparameters:")
        for model_name in self.models.keys():
            print(f"  - {model_name}")

        return self.models

    def save_results_to_csv(self, results, filename='Result/model_results.csv'):
        """Save model results to CSV"""
        os.makedirs('Result', exist_ok=True)

        rows = []
        for model_name, metrics in results.items():
            row = {'model': model_name}
            row.update(metrics)
            rows.append(row)

        df_results = pd.DataFrame(rows)
        df_results.to_csv(filename, index=False)
        print(f"\n[OK] Results saved to: {filename}")

        return df_results


# =============================================================================
# MAIN EXECUTION - Real Data Training
# =============================================================================

def train_with_real_data():
    """
    Train models with real DPM data
    Complete pipeline: Load -> Clean -> Feature Engineering -> WoE -> Train
    """
    print("=" * 80)
    print("DPM Model Training with Real Data (Clean Version - No W&B)")
    print("=" * 80)

    # 1. Initialize Pipeline
    print("\n[Step 1/8] Initialize Pipeline")
    pipeline = AdvancedDefaultPredictionPipeline(random_state=42)

    # 2. Load real data
    print("\n[Step 2/8] Load Real Data")
    df = pipeline.load_real_data('source/DPM_merged_cleaned.xlsx')

    # 3. Handle missing values
    print("\n[Step 3/8] Handle Missing Values")
    df = pipeline.handle_missing_values(df)

    # 4. Convert data types
    print("\n[Step 4/8] Convert Data Types")
    df = pipeline.convert_data_types(df)

    # 5. Define default label
    print("\n[Step 5/8] Define Default Label")
    df = pipeline.define_default(df)

    # 6. Feature engineering (credit-focused + Tier 1)
    print("\n[Step 6/8] Feature Engineering")
    df = pipeline.credit_focused_feature_engineering(df)

    # 7. Prepare modeling features
    print("\n[Step 7/8] Prepare Features")
    categorical_features, numerical_features = pipeline.prepare_features_for_modeling(df)

    # 8. Train/Test split
    print("\n[Step 8/8] Train/Test Split")

    # Remove row 0 (header reference) and non-feature columns
    df_model = df[df.index != 0].copy()

    all_features = categorical_features + numerical_features
    X = df_model[all_features]
    y = df_model['Default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train):,} samples (Default rate: {y_train.mean():.2%})")
    print(f"Test set: {len(X_test):,} samples (Default rate: {y_test.mean():.2%})")

    # 9. WoE Encoding (categorical features only)
    print("\n" + "=" * 80)
    print("WoE Encoding for Categorical Features")
    print("=" * 80)

    woe_encoder = WoEEncoder()
    X_train_woe = woe_encoder.fit_transform(
        X_train[categorical_features],
        y_train,
        categorical_features
    )
    X_test_woe = woe_encoder.transform(
        X_test[categorical_features],
        categorical_features
    )

    # 10. Combine WoE features and numerical features
    print("\n" + "=" * 80)
    print("Combine WoE and Numerical Features")
    print("=" * 80)

    woe_cols = [col for col in X_train_woe.columns if '_WoE' in col]

    X_train_final = pd.concat([
        X_train[numerical_features].reset_index(drop=True),
        X_train_woe[woe_cols].reset_index(drop=True)
    ], axis=1)

    X_test_final = pd.concat([
        X_test[numerical_features].reset_index(drop=True),
        X_test_woe[woe_cols].reset_index(drop=True)
    ], axis=1)

    print(f"\nFinal feature count: {X_train_final.shape[1]}")
    print(f"  - Numerical features: {len(numerical_features)}")
    print(f"  - WoE features: {len(woe_cols)}")

    # 11. Train models
    print("\n" + "=" * 80)
    print("Model Training")
    print("=" * 80)

    models = pipeline.initialize_models()
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train_final, y_train)

        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Default'])}")

        results[model_name] = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # 12. Best model
    print("\n" + "=" * 80)
    print("Best Model")
    print("=" * 80)

    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_auc = results[best_model_name]['auc']

    print(f"\nBest Model: {best_model_name}")
    print(f"AUC-ROC: {best_auc:.4f}")

    # 13. Stacking Ensemble
    print("\n" + "=" * 80)
    print("Stacking Ensemble Model")
    print("=" * 80)

    base_models = [
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM']),
        ('cat', models['CatBoost'])
    ]

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )

    print("\nTraining Stacking Ensemble...")
    stacking_model.fit(X_train_final, y_train)

    # Evaluate Stacking
    stack_pred = stacking_model.predict(X_test_final)
    stack_pred_proba = stacking_model.predict_proba(X_test_final)[:, 1]

    stack_auc = roc_auc_score(y_test, stack_pred_proba)
    stack_accuracy = accuracy_score(y_test, stack_pred)
    stack_precision = precision_score(y_test, stack_pred)
    stack_recall = recall_score(y_test, stack_pred)
    stack_f1 = f1_score(y_test, stack_pred)

    print(f"\nStacking Ensemble Performance:")
    print(f"  AUC-ROC: {stack_auc:.4f}")
    print(f"  Accuracy: {stack_accuracy:.4f}")
    print(f"  Precision: {stack_precision:.4f}")
    print(f"  Recall: {stack_recall:.4f}")
    print(f"  F1-Score: {stack_f1:.4f}")
    print(f"\n{classification_report(y_test, stack_pred, target_names=['Normal', 'Default'])}")

    results['Stacking'] = {
        'auc': stack_auc,
        'accuracy': stack_accuracy,
        'precision': stack_precision,
        'recall': stack_recall,
        'f1': stack_f1
    }

    # Update best model if stacking is better
    if stack_auc > best_auc:
        best_model_name = 'Stacking'
        best_auc = stack_auc
        print(f"\n[UPDATE] New best model: Stacking (AUC: {stack_auc:.4f})")

    # 14. Save results to CSV
    print("\n" + "=" * 80)
    print("Save Results")
    print("=" * 80)

    pipeline.save_results_to_csv(results, 'Result/model_results.csv')

    # 15. Save best model
    os.makedirs('models', exist_ok=True)
    if best_model_name == 'Stacking':
        joblib.dump(stacking_model, 'models/best_model_stacking.pkl')
        print(f"[OK] Best model (Stacking) saved to: models/best_model_stacking.pkl")
    else:
        joblib.dump(models[best_model_name], f'models/best_model_{best_model_name}.pkl')
        print(f"[OK] Best model ({best_model_name}) saved to: models/best_model_{best_model_name}.pkl")

    # 16. Save WoE encoder
    joblib.dump(woe_encoder, 'models/woe_encoder.pkl')
    print(f"[OK] WoE encoder saved to: models/woe_encoder.pkl")

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"\nResults saved to:")
    print(f"  - Result/model_results.csv")
    print(f"  - models/best_model_*.pkl")
    print(f"  - models/woe_encoder.pkl")

    return pipeline, results, woe_encoder


if __name__ == "__main__":
    # Run normal training (no W&B)
    pipeline, results, woe_encoder = train_with_real_data()
