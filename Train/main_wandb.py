# -*- coding: utf-8 -*-
"""
Default Prediction Model Training Pipeline
=====================================
Complete machine learning pipeline including:
- Advanced feature engineering
- Automated hyperparameter tuning
- Model ensemble
- SHAP analysis
- Model persistence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 載入機器學習相關套件
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, QuantileTransformer, 
    LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, log_loss
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer

# 進階模型
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# SHAP解釋性
import shap

# Weights & Biases for experiment tracking and hyperparameter tuning
import wandb

# 其他工具
import joblib
import json
import re
from datetime import datetime
import os
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sys
sys.path.append('..')
import config

# Import feature engineering from dedicated module
from feature_engineering import (
    OverduePatternEncoder,
    GeographicRiskEncoder
)

# W&B 實驗追蹤
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("W&B not available, logging disabled")
    WANDB_AVAILABLE = False
    wandb = None

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()  # 載入 .env 檔案
    print("Environment variables loaded successfully")
except ImportError:
    print("WARNING: python-dotenv not installed, using system environment variables")

# 設定matplotlib中文顯示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class WoEEncoder:
    """
    Weight of Evidence (WoE) 編碼器
    用於將分類變數轉換為 WoE 值
    """

    def __init__(self):
        self.woe_dict = {}
        self.iv_dict = {}

    def calculate_woe_iv(self, df, feature, target='Default'):
        """計算 WoE 和 IV 值"""
        grouped = df.groupby(feature)[target].agg(['sum', 'count'])
        grouped.columns = ['bad', 'total']
        grouped['good'] = grouped['total'] - grouped['bad']

        total_good = grouped['good'].sum()
        total_bad = grouped['bad'].sum()

        if total_good == 0 or total_bad == 0:
            print(f"[WARNING] {feature}: 沒有足夠的好壞樣本，跳過")
            return None

        grouped['good_pct'] = grouped['good'] / total_good
        grouped['bad_pct'] = grouped['bad'] / total_bad
        grouped['good_pct'] = grouped['good_pct'].replace(0, 0.0001)
        grouped['bad_pct'] = grouped['bad_pct'].replace(0, 0.0001)

        grouped['WoE'] = np.log(grouped['good_pct'] / grouped['bad_pct'])
        grouped['IV'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['WoE']

        return grouped.reset_index()

    def fit(self, X, y, categorical_features):
        """訓練 WoE 編碼器"""
        df = X.copy()
        df['Default'] = y

        print("=" * 70)
        print("WoE Encoding")
        print("=" * 70)

        for feature in categorical_features:
            if feature not in df.columns:
                print(f"[WARNING] {feature} not found, skip")
                continue

            woe_df = self.calculate_woe_iv(df, feature, 'Default')

            if woe_df is not None:
                self.woe_dict[feature] = dict(zip(woe_df[feature], woe_df['WoE']))
                total_iv = woe_df['IV'].sum()
                self.iv_dict[feature] = total_iv

                strength = ("Strong" if total_iv > 0.3 else
                          "Medium" if total_iv > 0.1 else
                          "Weak" if total_iv > 0.02 else "Very Weak")
                print(f"\n{feature}:")
                print(f"  IV = {total_iv:.4f} ({strength})")

        # 排序
        print("\n" + "=" * 70)
        print("Feature Importance (by IV)")
        print("=" * 70)
        sorted_iv = sorted(self.iv_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, iv in sorted_iv:
            print(f"{feature:40s} | IV = {iv:.4f}")

    def transform(self, X, categorical_features):
        """轉換為 WoE 值"""
        X_woe = X.copy()

        for feature in categorical_features:
            if feature not in self.woe_dict:
                continue

            woe_map = self.woe_dict[feature]
            X_woe[f'{feature}_WoE'] = X[feature].map(woe_map).fillna(0)

        return X_woe

    def fit_transform(self, X, y, categorical_features):
        """訓練並轉換"""
        self.fit(X, y, categorical_features)
        return self.transform(X, categorical_features)

    def get_important_features(self, iv_threshold=0.02):
        """根據 IV 值篩選重要特徵"""
        return [f for f, iv in self.iv_dict.items() if iv >= iv_threshold]


class TargetEncoder:
    """
    Target Encoding (Mean Encoding)
    將分類變數編碼為目標變數的平均值
    """
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.target_mean = {}
        self.global_mean = None

    def fit(self, X, y, categorical_features):
        """訓練 Target Encoder"""
        df = X.copy()
        df['target'] = y
        self.global_mean = y.mean()

        for feature in categorical_features:
            if feature not in df.columns:
                continue

            # Calculate mean target per category with smoothing
            agg = df.groupby(feature)['target'].agg(['mean', 'count'])

            # Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            smoothed_mean = (
                (agg['count'] * agg['mean'] + self.smoothing * self.global_mean) /
                (agg['count'] + self.smoothing)
            )

            self.target_mean[feature] = smoothed_mean.to_dict()

    def transform(self, X, categorical_features):
        """轉換為 Target Encoding 值"""
        X_encoded = X.copy()

        for feature in categorical_features:
            if feature not in self.target_mean:
                continue

            target_map = self.target_mean[feature]
            X_encoded[f'{feature}_target_enc'] = X[feature].map(target_map).fillna(self.global_mean)

        return X_encoded

    def fit_transform(self, X, y, categorical_features):
        self.fit(X, y, categorical_features)
        return self.transform(X, categorical_features)


class GeographicRiskEncoder:
    """
    Geographic Risk Feature Engineering
    計算地理位置的違約風險特徵
    """
    def __init__(self, min_samples=10):
        self.min_samples = min_samples
        self.res_risk_map = {}
        self.perm_risk_map = {}
        self.city_risk_map = {}
        self.global_default_rate = None

    def fit(self, X, y, res_col='post code of residential address',
            perm_col='post code of permanent address'):
        """訓練地理風險編碼器"""
        df = X.copy()
        df['Default'] = y
        self.global_default_rate = y.mean()

        print("\n" + "=" * 70)
        print("Geographic Risk Feature Engineering")
        print("=" * 70)

        # Residential postal code risk
        if res_col in df.columns:
            res_risk = df.groupby(res_col)['Default'].agg(['mean', 'count'])
            res_risk.columns = ['default_rate', 'sample_count']

            # Only trust postal codes with enough samples
            res_risk['risk_score'] = res_risk['default_rate']
            res_risk.loc[res_risk['sample_count'] < self.min_samples, 'risk_score'] = self.global_default_rate

            self.res_risk_map = res_risk['risk_score'].to_dict()

            print(f"\nResidential postal codes: {len(res_risk)} unique")
            print(f"  Risk range: {res_risk['risk_score'].min():.4f} ~ {res_risk['risk_score'].max():.4f}")

            # Top risk areas
            top_risk = res_risk.nlargest(5, 'risk_score')
            print(f"\n  Top 5 highest risk areas:")
            for idx, (postal, row) in enumerate(top_risk.iterrows(), 1):
                print(f"    {idx}. {postal}: {row['risk_score']:.2%} (n={row['sample_count']:.0f})")

        # Permanent postal code risk
        if perm_col in df.columns:
            perm_risk = df.groupby(perm_col)['Default'].agg(['mean', 'count'])
            perm_risk['risk_score'] = perm_risk['mean']
            perm_risk.loc[perm_risk['count'] < self.min_samples, 'risk_score'] = self.global_default_rate

            self.perm_risk_map = perm_risk['risk_score'].to_dict()

            print(f"\nPermanent postal codes: {len(perm_risk)} unique")

        # City level risk (extract first 3 digits)
        if res_col in df.columns:
            df['city'] = df[res_col].astype(str).str[:3]
            city_risk = df.groupby('city')['Default'].agg(['mean', 'count'])
            city_risk['risk_score'] = city_risk['mean']
            city_risk.loc[city_risk['count'] < self.min_samples * 5, 'risk_score'] = self.global_default_rate

            self.city_risk_map = city_risk['risk_score'].to_dict()

            print(f"\nCities: {len(city_risk)} unique")

    def transform(self, X, res_col='post code of residential address',
                  perm_col='post code of permanent address'):
        """轉換為地理風險特徵"""
        X_geo = X.copy()

        # Residential risk score
        if res_col in X.columns and self.res_risk_map:
            X_geo['res_risk_score'] = X[res_col].map(self.res_risk_map).fillna(self.global_default_rate)

            # Risk level categorization
            X_geo['res_risk_level'] = pd.cut(
                X_geo['res_risk_score'],
                bins=[0, 0.03, 0.06, 0.10, 1.0],
                labels=['low', 'medium', 'high', 'very_high']
            )

        # Permanent risk score
        if perm_col in X.columns and self.perm_risk_map:
            X_geo['perm_risk_score'] = X[perm_col].map(self.perm_risk_map).fillna(self.global_default_rate)

        # City risk score
        if res_col in X.columns and self.city_risk_map:
            X_geo['city'] = X[res_col].astype(str).str[:3]
            X_geo['city_risk_score'] = X_geo['city'].map(self.city_risk_map).fillna(self.global_default_rate)

        # Address stability risk (combined feature)
        if 'address_match' in X.columns and 'res_risk_score' in X_geo.columns:
            X_geo['address_stability_risk'] = (
                (1 - X['address_match']) * 0.3 + X_geo['res_risk_score'] * 0.7
            )

        return X_geo

    def fit_transform(self, X, y, res_col='post code of residential address',
                     perm_col='post code of permanent address'):
        self.fit(X, y, res_col, perm_col)
        return self.transform(X, res_col, perm_col)


class AdvancedDefaultPredictionPipeline:
    """
    Advanced Default Prediction Pipeline for client risk assessment
    Suitable for banks/financial institutions for client application risk evaluation
    """

    def __init__(self, random_state=42, n_jobs=-1, use_wandb=True, wandb_project="DPM"):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run = None

        self.models = {}
        self.feature_engineered_data = None
        self.feature_names = None
        self.target_col = 'Default'
        self.categorical_features = []
        self.numerical_features = []
        self.preprocessing_pipeline = None
        self.best_model = None
        self.ensemble_model = None
        self.feature_importance_df = None
        self.shap_explainers = {}

        # 設定隨機種子
        np.random.seed(random_state)

        # WoE 編碼器
        self.woe_encoder = None

        # 初始化W&B
        if self.use_wandb:
            self._init_wandb()

    def load_real_data(self, file_path='source/DPM_merged_cleaned.xlsx'):
        """
        載入真實的 DPM 資料

        Args:
            file_path: Excel 檔案路徑

        Returns:
            DataFrame: 載入的資料
        """
        print("=" * 70)
        print("載入真實資料")
        print("=" * 70)

        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"\n原始資料: {len(df):,} 筆, {len(df.columns)} 個欄位")

        return df

    def handle_missing_values(self, df):
        """
        處理缺失值

        Strategy:
        - 逾期欄位 → 填 0 (代表無逾期記錄)
        - 月薪 → 填中位數
        - 分類欄位 → 填眾數或 '其他'
        """
        print("\n" + "=" * 70)
        print("處理缺失值")
        print("=" * 70)

        df_clean = df.copy()

        # 找到正確的欄位名稱（處理換行符號）
        def find_col(keyword):
            matches = [col for col in df_clean.columns if keyword in str(col).lower()]
            return matches[0] if matches else None

        # 1. 逾期相關欄位 → 填 0
        print("\n[1/6] 逾期相關欄位")
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
                    print(f"  {col}: 填補 {before} 個缺失值 (填 0)")

        # 2. 月薪 → 填中位數
        print("\n[2/6] 月薪")
        salary_col = find_col('salary')
        if salary_col:
            df_clean[salary_col] = pd.to_numeric(df_clean[salary_col], errors='coerce')
            salary_median = df_clean[df_clean.index != 0][salary_col].median()
            before = df_clean[salary_col].isnull().sum()
            if before > 0:
                df_clean[salary_col] = df_clean[salary_col].fillna(salary_median)
                print(f"  month salary: 填補 {before} 個缺失值 (中位數: {salary_median:.2f})")

        # 3. 教育程度 → 填眾數
        print("\n[3/6] 教育程度")
        edu_col = find_col('education')
        if edu_col:
            edu_mode = df_clean[edu_col].mode()[0] if len(df_clean[edu_col].mode()) > 0 else '高中/職'
            before = df_clean[edu_col].isnull().sum()
            if before > 0:
                df_clean[edu_col] = df_clean[edu_col].fillna(edu_mode)
                print(f"  education: 填補 {before} 個缺失值 (眾數: {edu_mode})")

        # 4. 居住狀況 → 填眾數
        print("\n[4/6] 居住狀況")
        residence_col = find_col('residence status')
        if residence_col:
            residence_mode = df_clean[residence_col].mode()[0] if len(df_clean[residence_col].mode()) > 0 else '租賃'
            before = df_clean[residence_col].isnull().sum()
            if before > 0:
                df_clean[residence_col] = df_clean[residence_col].fillna(residence_mode)
                print(f"  residence status: 填補 {before} 個缺失值 (眾數: {residence_mode})")

        # 5. 主要業務 → 填 '其他'
        print("\n[5/6] 主要經營業務")
        business_col = find_col('main business')
        if business_col:
            before = df_clean[business_col].isnull().sum()
            if before > 0:
                df_clean[business_col] = df_clean[business_col].fillna('其他')
                print(f"  main business: 填補 {before} 個缺失值 (填 '其他')")

        # 6. 其他欄位
        print("\n[6/6] 其他欄位")

        # 年資 → 中位數
        tenure_col = find_col('job tenure')
        if tenure_col:
            df_clean[tenure_col] = pd.to_numeric(df_clean[tenure_col], errors='coerce')
            tenure_median = df_clean[tenure_col].median()
            before = df_clean[tenure_col].isnull().sum()
            if before > 0:
                df_clean[tenure_col] = df_clean[tenure_col].fillna(tenure_median)
                print(f"  job tenure: 填補 {before} 個缺失值 (中位數: {tenure_median:.1f})")

        # 婚姻 → 眾數
        marriage_col = find_col('marriage')
        if marriage_col:
            marriage_mode = df_clean[marriage_col].mode()[0] if len(df_clean[marriage_col].mode()) > 0 else '未婚'
            before = df_clean[marriage_col].isnull().sum()
            if before > 0:
                df_clean[marriage_col] = df_clean[marriage_col].fillna(marriage_mode)
                print(f"  marriage: 填補 {before} 個缺失值 (眾數: {marriage_mode})")

        print("\n缺失值處理完成！")
        return df_clean

    def convert_data_types(self, df):
        """轉換資料型態"""

        print("\n" + "=" * 70)
        print("資料型態轉換")
        print("=" * 70)

        df_typed = df.copy()

        # 數值欄位
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

        print("\n數值欄位轉換:")
        for col in numeric_cols:
            if col in df_typed.columns:
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
                print(f"  {col}")

        # 分類欄位確保為字串
        categorical_cols = [
            'overdue status', 'education', 'residence status', 'main business',
            'product', 'marriage', 'post code of residential address',
            'post code of permanent address',
        ]

        print("\n分類欄位轉換:")
        for col in categorical_cols:
            if col in df_typed.columns:
                df_typed[col] = df_typed[col].astype(str)
                print(f"  {col}")

        return df_typed

    def define_default(self, df):
        """
        定義違約標籤

        Rule: M1+ (逾期30天以上) = 違約
        Updated: 2025-10-21 - Changed from M2+ to M1+ for better class balance
        """
        print("\n" + "=" * 70)
        print("定義違約標籤 (M1+ Definition)")
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

        print(f"\n總樣本數: {len(df):,}")
        print(f"違約樣本: {df['Default'].sum():,} ({df['Default'].mean():.2%})")
        print(f"正常樣本: {(df['Default']==0).sum():,} ({(df['Default']==0).mean():.2%})")

        return df

    def credit_focused_feature_engineering(self, df):
        """
        特徵工程 - 以徵審重點為導向 + DTI Focus

        徵審關注點:
        1. 還款能力 (財務能力)
        2. 聯絡穩定性 (找得到人)
        3. DTI (債務收入比) - CRITICAL
        """
        print("\n" + "=" * 70)
        print("Feature Engineering (Credit Assessment Focus + DTI)")
        print("=" * 70)

        df_fe = df.copy()

        # === 財務能力指標 ===
        print("\n[1/4] Financial Capability Indicators")

        # 1. 還款進度比率
        df_fe['payment_progress_ratio'] = (
            df_fe['paid installments'] / df_fe['loan term']
        ).fillna(0).clip(0, 1)
        print("  + payment_progress_ratio (paid/total installments)")

        # 2. 工作穩定性 (年資 >= 1年)
        df_fe['job_stable'] = (df_fe['job tenure'] >= 1).astype(int)
        print("  + job_stable (tenure >= 1 year)")

        # === 聯絡穩定性指標 ===
        print("\n[2/4] Contact Stability Indicators")

        # 3. 戶籍居住地一致性
        df_fe['address_match'] = (
            df_fe['post code of permanent address'] ==
            df_fe['post code of residential address']
        ).astype(int)
        print("  + address_match (permanent = residential)")

        # 4. 居住穩定性 (自有 > 租賃)
        stable_residence = ['自有', '配偶名下', '親友名下']
        df_fe['residence_stable'] = (
            df_fe['residence status'].isin(stable_residence)
        ).astype(int)
        print("  + residence_stable (own/spouse/family)")

        # === DTI & Financial Burden - CRITICAL! ===
        print("\n[3/4] DTI & Financial Burden (CRITICAL!)")

        # 5. DTI ratio
        if 'debt_to_income_ratio' in df_fe.columns:
            df_fe['dti_ratio'] = pd.to_numeric(df_fe['debt_to_income_ratio'], errors='coerce').fillna(0).clip(0, 2)
            print("  + dti_ratio (debt/income ratio - PRIMARY INDICATOR)")

        # 6. Payment pressure
        if 'payment_to_income_ratio' in df_fe.columns:
            df_fe['payment_pressure'] = pd.to_numeric(df_fe['payment_to_income_ratio'], errors='coerce').fillna(0).clip(0, 1)
            print("  + payment_pressure (payment/income ratio)")

        # === 逾期行為指標 (早期) ===
        print("\n[4/4] Early Overdue Behavior Indicators")

        # 7. 早期逾期次數
        overdue_cols = [
            'number of overdue before the first month',
            'number of overdue in the first half of the first month',
            'number of overdue in the second half of the first month',
        ]
        df_fe['early_overdue_count'] = df_fe[overdue_cols].sum(axis=1)
        print("  + early_overdue_count (sum of early overdue)")

        # 8. 是否有逾期記錄
        df_fe['has_overdue'] = (df_fe['early_overdue_count'] > 0).astype(int)
        print("  + has_overdue (binary flag)")

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
        """
        準備建模特徵 - 使用 feature_engineering.py 的統一特徵列表

        Returns:
            categorical_features: 需要 WoE 編碼的分類特徵
            numerical_features: 數值特徵
        """
        from feature_engineering import get_feature_lists

        print("\n" + "=" * 70)
        print("Prepare Features for Modeling")
        print("=" * 70)

        # 使用統一的特徵列表（包含 Tier 1 特徵）
        categorical_features, numerical_features, _ = get_feature_lists()

        # 過濾實際存在的欄位
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

    def advanced_feature_engineering(self, data):
        """
        進階特徵工程
        """
        print("進行進階特徵工程...")
        
        data_fe = data.copy()
        
        # === 識別特徵類型 ===
        self.categorical_features = [
            'Education_Level', 'Job_Category', 'Employment_Status', 
            'Marital_Status', 'Housing_Status', 'Has_Mortgage', 
            'Has_Car_Loan', 'Is_Holiday_Season', 'Loan_Origination_Month'
        ]
        
        self.numerical_features = [col for col in data_fe.columns 
                                 if col not in self.categorical_features + ['Customer_ID', 'Default']]
        
        # === 數值特徵的衍生特徵 ===
        # 1. 比率和交互特徵
        data_fe['Income_per_Age'] = data_fe['Annual_Income'] / data_fe['Age']
        data_fe['Loan_Amount_per_Credit_Limit'] = data_fe['Loan_Amount'] / (data_fe['Total_Credit_Limit'] + 1)
        data_fe['Payment_Burden_Ratio'] = (data_fe['Loan_Amount'] / data_fe['Loan_Term_Months']) / data_fe['Monthly_Income']
        
        # 2. 風險綜合指標
        data_fe['Payment_Risk_Score'] = (
            data_fe['Late_Payments_Count'] * 0.3 + 
            data_fe['Missed_Payments_Count'] * 0.7 +
            (1 - data_fe['On_Time_Payment_Rate']) * 0.5
        )
        
        data_fe['Financial_Stability_Score'] = (
            data_fe['Savings_to_Income_Ratio'] * 0.4 +
            (data_fe['Avg_Monthly_Balance'] / data_fe['Monthly_Income']).clip(0, 5) * 0.3 +
            (1 - data_fe['Credit_Utilization_Rate']) * 0.3
        )
        
        # 3. 經驗指標
        data_fe['Credit_Experience_Score'] = (
            np.log1p(data_fe['Credit_History_Months']) * 0.4 +
            np.log1p(data_fe['Previous_Loans_Count']) * 0.6
        )
        
        # === 分箱處理 (針對重要的連續變數) ===
        # 年齡分組
        data_fe['Age_Group'] = pd.cut(data_fe['Age'], 
                                    bins=[0, 25, 35, 45, 55, 100], 
                                    labels=['Young', 'Young_Adult', 'Middle_Age', 'Mature', 'Senior'])
        
        # 收入分組
        data_fe['Income_Quintile'] = pd.qcut(data_fe['Annual_Income'], 
                                           q=5, 
                                           labels=['Low', 'Lower_Mid', 'Mid', 'Upper_Mid', 'High'])
        
        # 貸款金額相對分組
        data_fe['Loan_Size_Category'] = pd.qcut(data_fe['Loan_Amount'], 
                                              q=3, 
                                              labels=['Small', 'Medium', 'Large'])
        
        # === 異常值處理標記 ===
        # 使用IQR方法識別異常值
        for col in ['Annual_Income', 'Loan_Amount', 'Debt_to_Income_Ratio']:
            Q1 = data_fe[col].quantile(0.25)
            Q3 = data_fe[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data_fe[f'{col}_Outlier'] = ((data_fe[col] < lower_bound) | 
                                       (data_fe[col] > upper_bound)).astype(int)
        
        # === 時間特徵擴展 ===
        data_fe['Loan_Month_Sin'] = np.sin(2 * np.pi * data_fe['Loan_Origination_Month'] / 12)
        data_fe['Loan_Month_Cos'] = np.cos(2 * np.pi * data_fe['Loan_Origination_Month'] / 12)
        
        # === 統計特徵 ===
        # 客戶風險群組統計
        risk_stats = data_fe.groupby(['Job_Category', 'Education_Level'])['Default'].agg(['mean', 'count']).reset_index()
        risk_stats.columns = ['Job_Category', 'Education_Level', 'Group_Default_Rate', 'Group_Size']
        data_fe = data_fe.merge(risk_stats, on=['Job_Category', 'Education_Level'], how='left')
        
        # 處理新分類變數
        new_categorical = ['Age_Group', 'Income_Quintile', 'Loan_Size_Category']
        self.categorical_features.extend(new_categorical)
        
        # 更新數值特徵列表
        self.numerical_features = [col for col in data_fe.columns 
                                 if col not in self.categorical_features + ['Customer_ID', 'Default']]
        
        print(f"特徵工程完成:")
        print(f"- 總特徵數: {len(self.categorical_features) + len(self.numerical_features)}")
        print(f"- 分類特徵數: {len(self.categorical_features)}")
        print(f"- 數值特徵數: {len(self.numerical_features)}")
        
        self.feature_engineered_data = data_fe
        return data_fe
    
    def create_preprocessing_pipeline(self):
        """
        創建資料預處理Pipeline
        """
        # 數值特徵預處理
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),  # 使用KNN填補缺失值
            ('scaler', RobustScaler())  # 使用RobustScaler對異常值更穩健
        ])
        
        # 分類特徵預處理
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # 組合預處理器
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return self.preprocessing_pipeline
    
    def advanced_train_test_split(self, data, test_size=0.2, val_size=0.1):
        """
        進階的訓練/驗證/測試集分割
        包含時間序列考量和分層抽樣
        """
        # 排除ID和目標變數
        features = [col for col in data.columns if col not in ['Customer_ID', 'Default']]
        X = data[features]
        y = data['Default']
        
        # 第一次分割：分出測試集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # 第二次分割：從剩餘數據中分出驗證集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"資料分割完成:")
        print(f"- 訓練集: {len(X_train):,} 樣本 (違約率: {y_train.mean():.2%})")
        print(f"- 驗證集: {len(X_val):,} 樣本 (違約率: {y_val.mean():.2%})")
        print(f"- 測試集: {len(X_test):,} 樣本 (違約率: {y_test.mean():.2%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """
        處理類別不平衡
        """
        print(f"處理類別不平衡 (方法: {method})...")
        print(f"原始分佈 - 正常: {(~y_train.astype(bool)).sum():,}, 違約: {y_train.sum():,}")
        
        if method == 'smote':
            # SMOTE過採樣
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif method == 'undersample':
            # 隨機欠採樣
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        
        print(f"重採樣後分佈 - 正常: {(~y_resampled.astype(bool)).sum():,}, 違約: {y_resampled.sum():,}")
        return X_resampled, y_resampled
    
    def initialize_models(self, use_best_params=False, best_params_path='Train/best_params.json'):
        """
        初始化多個模型，使用config中的預設參數或WandB最佳參數

        Args:
            use_best_params: 是否使用 WandB Sweep 的最佳超參數
            best_params_path: 最佳超參數 JSON 檔案路徑
        """
        # 決定使用哪一組參數
        if use_best_params and os.path.exists(best_params_path):
            print("\n" + "=" * 70)
            print("Loading Best Parameters from WandB")
            print("=" * 70)

            try:
                with open(best_params_path, 'r', encoding='utf-8') as f:
                    best_params = json.load(f)

                print(f"[OK] Loaded best parameters from: {best_params['metadata']['run_name']}")
                print(f"     Best val_auc: {best_params['metadata']['val_auc']:.4f}")

                model_params = {
                    'XGBoost': best_params['XGBoost'],
                    'LightGBM': best_params['LightGBM'],
                    'CatBoost': best_params['CatBoost']
                }

            except Exception as e:
                print(f"[WARNING] Failed to load {best_params_path}: {e}")
                print(f"[INFO] Using config.py parameters")
                model_params = config.MODEL_PARAMS

        else:
            if use_best_params:
                print(f"\n[WARNING] {best_params_path} not found")
                print(f"[INFO] Run 'python main_wandb.py --fetch-best' first")
            print(f"[INFO] Using default parameters from config.py")
            model_params = config.MODEL_PARAMS

        # 確保必要的參數存在
        for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            if 'random_state' not in model_params[model_name]:
                model_params[model_name]['random_state'] = self.random_state
            if model_name == 'XGBoost' and 'eval_metric' not in model_params[model_name]:
                model_params[model_name]['eval_metric'] = 'auc'
            if model_name == 'XGBoost' and 'verbosity' not in model_params[model_name]:
                model_params[model_name]['verbosity'] = 0
            if model_name == 'LightGBM' and 'verbosity' not in model_params[model_name]:
                model_params[model_name]['verbosity'] = -1
            if model_name == 'CatBoost' and 'verbose' not in model_params[model_name]:
                model_params[model_name]['verbose'] = False

        self.models = {
            'XGBoost': xgb.XGBClassifier(**model_params['XGBoost']),
            'LightGBM': lgb.LGBMClassifier(**model_params['LightGBM']),
            'CatBoost': CatBoostClassifier(**model_params['CatBoost'])
        }

        return self.models
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, cv_folds=5):
        """
        超參數調優
        """
        print("開始超參數調優...")
        
        # 定義搜索空間
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_samples': [10, 20, 30]
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [32, 64, 128]
            }
        }
        
        best_models = {}
        cv_results = {}
        
        # 分層交叉驗證
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            print(f"\n調優 {model_name}...")
            
            # 使用RandomizedSearchCV以提高效率
            random_search = RandomizedSearchCV(
                model,
                param_grids[model_name],
                cv=skf,
                scoring='roc_auc',
                n_iter=20,  # 隨機搜索20組參數
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
            
            random_search.fit(X_train, y_train)
            
            best_models[model_name] = random_search.best_estimator_
            cv_results[model_name] = {
                'best_score': random_search.best_score_,
                'best_params': random_search.best_params_
            }
            
            print(f"{model_name} 最佳CV分數: {random_search.best_score_:.4f}")
            print(f"{model_name} 最佳參數: {random_search.best_params_}")
        
        self.models = best_models
        return cv_results
    
    def train_and_evaluate_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        訓練和評估所有模型
        """
        results = {}
        
        print("\n開始模型訓練和評估...")
        
        for model_name, model in self.models.items():
            print(f"\n訓練 {model_name}...")
            
            # 訓練模型
            if model_name in ['XGBoost', 'LightGBM']:
                # 對於支持early stopping的模型
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # 在各個數據集上進行預測
            train_pred = model.predict(X_train)
            train_pred_proba = model.predict_proba(X_train)[:, 1]
            
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            test_pred = model.predict(X_test)
            test_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 計算評估指標
            results[model_name] = {
                'train': self._calculate_metrics(y_train, train_pred, train_pred_proba),
                'val': self._calculate_metrics(y_val, val_pred, val_pred_proba),
                'test': self._calculate_metrics(y_test, test_pred, test_pred_proba)
            }
            
            # 印出結果
            self._print_model_results(model_name, results[model_name])
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """計算評估指標"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        }
    
    def _print_model_results(self, model_name, results):
        """印出模型評估結果"""
        for dataset in ['train', 'val', 'test']:
            print(f"\n{model_name} - {dataset.upper()} Results:")
            metrics = results[dataset]
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    def build_ensemble(self):
        """建構ensemble模型"""
        print("\n建構ensemble模型...")
        
        # 創建voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'
        )
        
        return self.ensemble_model
    
    def analyze_feature_importance(self, X):
        """分析特徵重要性"""
        print("\n分析特徵重要性...")
        
        importance_dict = {}
        feature_names = (self.numerical_features + 
                        [f"{feat}_{val}" for feat in self.categorical_features
                         for val in range(3)])  # Assuming max 3 categories after one-hot
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        # Create DataFrame
        self.feature_importance_df = pd.DataFrame(importance_dict, index=feature_names)
        return self.feature_importance_df
    
    def shap_analysis(self, X_sample):
        """SHAP值分析"""
        print("\n執行SHAP分析...")
        
        for model_name, model in self.models.items():
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            self.shap_explainers[model_name] = {
                'explainer': explainer,
                'values': shap_values
            }
        
        return self.shap_explainers
    
    def save_model(self, path='models'):
        """保存模型和相關資訊"""
        print("\n保存模型...")
        
        # 創建模型目錄
        os.makedirs(path, exist_ok=True)
        
        # 保存每個模型
        for model_name, model in self.models.items():
            model_path = os.path.join(path, f'{model_name}.pkl')
            joblib.dump(model, model_path)
        
        # 保存ensemble模型
        if self.ensemble_model is not None:
            ensemble_path = os.path.join(path, 'ensemble_model.pkl')
            joblib.dump(self.ensemble_model, ensemble_path)
        
        # 保存特徵重要性
        if self.feature_importance_df is not None:
            importance_path = os.path.join(path, 'feature_importance.csv')
            self.feature_importance_df.to_csv(importance_path)
        
        # 保存預處理pipeline
        if self.preprocessing_pipeline is not None:
            pipeline_path = os.path.join(path, 'preprocessing_pipeline.pkl')
            joblib.dump(self.preprocessing_pipeline, pipeline_path)
        
        print(f"模型和相關資訊已保存至 {path} 目錄")


    def predict_client_default(self, client_data):
        """
        Predict client default risk for risk assessment

        Args:
            client_data (dict): Client data

        Returns:
            dict: Default risk prediction results
        """
        # 驗證客戶資料
        is_valid, errors = config.validate_client_data(client_data)
        if not is_valid:
            return {
                'status': 'error',
                'errors': errors,
                'recommendation': 'Data validation failed'
            }

        # 確保模型已訓練
        if not self.models or not self.preprocessing_pipeline:
            raise ValueError("模型尚未訓練，請先調用訓練方法")

        # 準備資料
        df = pd.DataFrame([client_data])

        # 特徵工程 (簡化版本，針對單一預測)
        if 'Debt_to_Income_Ratio' not in df.columns:
            df['Debt_to_Income_Ratio'] = df.get('Debt', 0) / df.get('Annual_Income', 1)

        # 預處理
        try:
            # 選擇模型預期的特徵
            feature_columns = [col for col in df.columns
                             if col in self.numerical_features + self.categorical_features]
            X = df[feature_columns]

            # 應用預處理
            X_processed = self.preprocessing_pipeline.transform(X)

            # 獲取所有模型的預測
            model_predictions = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(X_processed)[0, 1]
                model_predictions[model_name] = pred_proba

            # 使用ensemble模型預測 (如果可用)
            if self.ensemble_model:
                ensemble_prob = self.ensemble_model.predict_proba(X_processed)[0, 1]
                final_probability = ensemble_prob
            else:
                # 使用平均預測
                final_probability = np.mean(list(model_predictions.values()))

            # 風險分類 (好/壞客戶分群)
            risk_category = 'Good' if final_probability < 0.5 else 'Bad'

            # 組合結果
            result = {
                'status': 'success',
                'client_id': client_data.get('Client_ID', 'UNKNOWN'),
                'model_predictions': model_predictions,
                'final_probability': final_probability,
                'risk_category': risk_category,
                'risk_score': int((1 - final_probability) * 100)  # 0-100分數，越高越好
            }

            return result

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'recommendation': 'Prediction failed - manual review required'
            }

    def batch_client_scoring(self, clients_df):
        """
        Batch client scoring for risk assessment

        Args:
            clients_df (DataFrame): Client data DataFrame

        Returns:
            DataFrame: DataFrame with scoring results
        """
        results = []

        for idx, row in clients_df.iterrows():
            client_data = row.to_dict()
            prediction = self.predict_client_default(client_data)

            if prediction['status'] == 'success':
                result_row = {
                    'Client_ID': client_data.get('Client_ID', idx),
                    'Risk_Score': prediction['risk_score'],
                    'Risk_Category': prediction['risk_category'],
                    'Default_Probability': prediction['final_probability'],
                    'XGBoost_Prob': prediction['model_predictions'].get('XGBoost', 0),
                    'LightGBM_Prob': prediction['model_predictions'].get('LightGBM', 0),
                    'CatBoost_Prob': prediction['model_predictions'].get('CatBoost', 0)
                }
            else:
                result_row = {
                    'Client_ID': client_data.get('Client_ID', idx),
                    'Risk_Score': 0,
                    'Risk_Category': 'ERROR',
                    'Default_Probability': 1.0,
                    'XGBoost_Prob': 1.0,
                    'LightGBM_Prob': 1.0,
                    'CatBoost_Prob': 1.0
                }

            results.append(result_row)

        return pd.DataFrame(results)

    def generate_model_report(self, results):
        """
        生成模型報告

        Args:
            results (dict): 模型評估結果

        Returns:
            dict: 詳細的模型報告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'business_metrics': {},
            'recommendations': []
        }

        # 模型性能分析
        for model_name, model_results in results.items():
            test_metrics = model_results['test']
            report['model_performance'][model_name] = {
                'auc_roc': test_metrics['auc_roc'],
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1'],
                'log_loss': test_metrics['log_loss']
            }

        # 找出最佳模型
        best_model = max(results.keys(),
                        key=lambda x: results[x]['test']['auc_roc'])
        report['best_model'] = best_model
        report['best_auc'] = results[best_model]['test']['auc_roc']

        # 業務指標建議
        if report['best_auc'] >= 0.85:
            report['recommendations'].append("模型性能優秀，可以部署到生產環境")
        elif report['best_auc'] >= 0.75:
            report['recommendations'].append("模型性能良好，建議進一步調優")
        else:
            report['recommendations'].append("模型性能需要改善，建議重新訓練")

        return report

    def _init_wandb(self):
        """初始化W&B實驗追蹤"""
        if not self.use_wandb:
            return

        try:
            # 從環境變數取得 API key
            api_key = os.getenv('WANDB_API_KEY')
            project_name = os.getenv('WANDB_PROJECT_NAME', 'dpm_run')
            entity = os.getenv('WANDB_ENTITY')

            if api_key and api_key != 'your_wandb_api_key_here':
                os.environ['WANDB_API_KEY'] = api_key
                print(f"[OK] W&B API key loaded from .env")
            else:
                print("[WARNING] W&B API key not set in .env file, please update .env")
                print("   You can get your API key from: https://wandb.ai/settings")
                return

            # 初始化 W&B
            init_kwargs = {
                'project': project_name,
                'name': f"dpm_real_data_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'config': {
                    'random_state': self.random_state,
                    'model_type': 'ensemble',
                    'data_version': 'v1.0_real_data',
                    'encoding_method': 'WoE',
                },
                'tags': ['real_data', 'credit_risk', 'woe_encoding', 'wandb_tracking'],
                'mode': os.getenv('WANDB_MODE', 'online')
            }

            # Only add entity if it's set (avoid permission issues)
            if entity:
                init_kwargs['entity'] = entity

            wandb.init(**init_kwargs)
            self.wandb_run = wandb.run
            print(f"[OK] W&B initialized: {self.wandb_run.name}")
            print(f"   Project: {project_name}")
            if entity:
                print(f"   Entity: {entity}")
        except Exception as e:
            print(f"[ERROR] W&B initialization failed: {e}")
            print("   Please check your .env file and W&B API key")
            self.use_wandb = False

    def log_to_wandb(self, metrics, step=None, **kwargs):
        """記錄指標到W&B"""
        if not self.use_wandb or not self.wandb_run:
            return

        log_data = {**metrics, **kwargs}
        if step:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)


# =============================================================================
# MAIN EXECUTION - Real Data Training
# =============================================================================

def train_with_real_data(use_best_params=False, best_params_path='Train/best_params.json'):
    """
    使用真實 DPM 資料訓練模型
    完整流程：載入 → 清理 → 特徵工程 → WoE 編碼 → 訓練

    Args:
        use_best_params: 是否使用 WandB Sweep 的最佳超參數
        best_params_path: 最佳超參數 JSON 檔案路徑
    """
    print("=" * 80)
    print("DPM Model Training with Real Data")
    print("=" * 80)

    if use_best_params:
        print("\n[MODE] Using BEST PARAMETERS from WandB Sweep")
    else:
        print("\n[MODE] Using DEFAULT PARAMETERS from config.py")

    # 1. 初始化 Pipeline
    print("\n[Step 1/8] Initialize Pipeline")
    pipeline = AdvancedDefaultPredictionPipeline(random_state=42)

    # 2. 載入真實資料
    print("\n[Step 2/8] Load Real Data")
    df = pipeline.load_real_data('source/DPM_merged_cleaned.xlsx')

    # 3. 處理缺失值
    print("\n[Step 3/8] Handle Missing Values")
    df = pipeline.handle_missing_values(df)

    # 4. 轉換資料型態
    print("\n[Step 4/8] Convert Data Types")
    df = pipeline.convert_data_types(df)

    # 5. 定義違約標籤
    print("\n[Step 5/8] Define Default Label")
    df = pipeline.define_default(df)

    # 6. 特徵工程（徵審導向）
    print("\n[Step 6/8] Feature Engineering")
    df = pipeline.credit_focused_feature_engineering(df)

    # 7. 準備建模特徵
    print("\n[Step 7/8] Prepare Features")
    categorical_features, numerical_features = pipeline.prepare_features_for_modeling(df)

    # 8. 訓練/測試分割
    print("\n[Step 8/8] Train/Test Split")

    # 排除第0筆（標題對照）和非特徵欄位
    df_model = df[df.index != 0].copy()

    all_features = categorical_features + numerical_features
    X = df_model[all_features]
    y = df_model['Default']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train):,} samples (Default rate: {y_train.mean():.2%})")
    print(f"Test set: {len(X_test):,} samples (Default rate: {y_test.mean():.2%})")

    # 9. WoE 編碼（只對分類特徵）
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

    # 10. 合併 WoE 特徵和數值特徵
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

    # 11. 訓練模型
    print("\n" + "=" * 80)
    print("Model Training")
    print("=" * 80)

    models = pipeline.initialize_models(use_best_params=use_best_params,
                                       best_params_path=best_params_path)
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Define custom x-axis for this model's training metrics
        if pipeline.use_wandb and pipeline.wandb_run:
            wandb.define_metric(f"{model_name}/epoch")
            wandb.define_metric(f"{model_name}/train/*", step_metric=f"{model_name}/epoch")

        model.fit(X_train_final, y_train)

        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final)[:, 1]

        from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
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
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        # Log to W&B
        if pipeline.use_wandb and pipeline.wandb_run:
            wandb_metrics = {
                f"{model_name}/test/auc_roc": auc,
                f"{model_name}/test/accuracy": accuracy,
                f"{model_name}/test/precision": precision,
                f"{model_name}/test/recall": recall,
                f"{model_name}/test/f1": f1
            }
            pipeline.log_to_wandb(wandb_metrics)

            # Log confusion matrix
            y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
            y_pred_array = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)

            wandb.log({
                f"{model_name}/confusion_matrix_default_threshold": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_test_array.tolist(),
                    preds=y_pred_array.tolist(),
                    class_names=['Normal', 'Default'],
                    title=f"{model_name} Confusion Matrix (Threshold=0.5)"
                )
            })

    # 12. 最佳模型
    print("\n" + "=" * 80)
    print("Best Model")
    print("=" * 80)

    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_auc = results[best_model_name]['auc']

    print(f"\nBest Model: {best_model_name}")
    print(f"AUC-ROC: {best_auc:.4f}")

    # Log best model to W&B
    if pipeline.use_wandb and pipeline.wandb_run:
        wandb.log({
            "best_model": best_model_name,
            "best_auc_roc": best_auc
        })

    # 13. SHAP Analysis for Feature Importance (includes DTI!)
    print("\n" + "=" * 80)
    print("SHAP Feature Importance Analysis")
    print("=" * 80)

    # Sample for SHAP (use subset for speed)
    sample_size = min(500, len(X_test_final))
    X_shap_sample = X_test_final.sample(n=sample_size, random_state=42)

    shap_importance_dict = {}

    for model_name, model_result in results.items():
        model = model_result['model']
        print(f"\n[SHAP] Analyzing {model_name}...")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap_sample)

            # For binary classification, take positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance_dict[model_name] = mean_abs_shap

            print(f"  SHAP analysis completed for {model_name}")

        except Exception as e:
            print(f"  [WARNING] SHAP failed for {model_name}: {e}")

    # Combine SHAP importance across models
    if shap_importance_dict:
        feature_names = X_train_final.columns.tolist()
        shap_df = pd.DataFrame(shap_importance_dict, index=feature_names)
        shap_df['Mean_SHAP'] = shap_df.mean(axis=1)
        shap_df = shap_df.sort_values('Mean_SHAP', ascending=False)

        print(f"\n{'='*80}")
        print("Top 15 Features by SHAP Importance")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Feature':<35} {'SHAP Importance':<20}")
        print(f"{'-'*80}")

        for rank, (feat, row) in enumerate(shap_df.head(15).iterrows(), 1):
            print(f"{rank:<6} {feat:<35} {row['Mean_SHAP']:<20.6f}")

        # Check if DTI features are in top features
        dti_features_check = [f for f in feature_names if 'dti' in f.lower() or 'payment_pressure' in f.lower()]
        if dti_features_check:
            print(f"\n{'='*80}")
            print("DTI-Related Features in SHAP Analysis:")
            print(f"{'='*80}")
            for feat in dti_features_check:
                if feat in shap_df.index:
                    shap_val = shap_df.loc[feat, 'Mean_SHAP']
                    rank = (shap_df['Mean_SHAP'] >= shap_val).sum()
                    print(f"  {feat:<35} | SHAP: {shap_val:.6f} | Rank: #{rank}")
        else:
            print(f"\n[WARNING] DTI features not found in feature list!")

        # Save SHAP report
        shap_df.to_csv('Result/SHAP_Feature_Importance.csv')
        print(f"\n[OK] SHAP report saved to Result/SHAP_Feature_Importance.csv")

        # Log to W&B
        if pipeline.use_wandb and pipeline.wandb_run:
            shap_table_data = [[feat, row['Mean_SHAP']] for feat, row in shap_df.head(20).iterrows()]
            wandb.log({
                "feature_importance/shap_values": wandb.Table(
                    data=shap_table_data,
                    columns=["Feature", "Mean_Abs_SHAP"]
                )
            })

    # 14. IV Feature Importance (for categorical features)
    print("\n" + "=" * 80)
    print("IV Feature Importance (Categorical Features)")
    print("=" * 80)

    important_features = woe_encoder.get_important_features(iv_threshold=0.02)
    print(f"\nImportant features (IV > 0.02): {len(important_features)}")
    for feat in important_features:
        iv = woe_encoder.iv_dict[feat]
        print(f"  {feat:40s} | IV = {iv:.4f}")

    # Log feature importance to W&B
    if pipeline.use_wandb and pipeline.wandb_run:
        iv_table_data = [[feat, woe_encoder.iv_dict[feat]] for feat in important_features]
        wandb.log({
            "feature_importance/iv_values": wandb.Table(
                data=iv_table_data,
                columns=["Feature", "IV_Value"]
            )
        })

    # 15. Threshold Optimization for Better Recall
    print("\n" + "=" * 80)
    print("Threshold Optimization (Option A: Improve Recall)")
    print("=" * 80)

    def find_optimal_threshold(y_true, y_pred_proba, metric='f1', beta=2):
        """
        Find optimal classification threshold

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: 'f1', 'f2' (recall-focused), or 'balanced' (equal weight)
            beta: For F-beta score (beta=2 weights recall 2x higher than precision)
        """
        from sklearn.metrics import fbeta_score

        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []

        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred_thresh)
            elif metric == 'f2':
                score = fbeta_score(y_true, y_pred_thresh, beta=beta)
            elif metric == 'balanced':
                prec = precision_score(y_true, y_pred_thresh)
                rec = recall_score(y_true, y_pred_thresh)
                score = (prec + rec) / 2

            scores.append(score)

        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]

        return optimal_threshold, optimal_score

    # Find optimal thresholds for each model
    print("\nFinding optimal thresholds for better recall (F2-score)...")
    optimal_thresholds = {}

    for model_name, model_result in results.items():
        y_pred_proba = model_result['probabilities']

        # F2 score (weights recall 2x more than precision)
        optimal_thresh, optimal_f2 = find_optimal_threshold(y_test, y_pred_proba, metric='f2', beta=2)

        # Evaluate at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_thresh).astype(int)

        optimal_accuracy = accuracy_score(y_test, y_pred_optimal)
        optimal_precision = precision_score(y_test, y_pred_optimal)
        optimal_recall = recall_score(y_test, y_pred_optimal)
        optimal_f1 = f1_score(y_test, y_pred_optimal)

        optimal_thresholds[model_name] = {
            'threshold': optimal_thresh,
            'f2_score': optimal_f2,
            'accuracy': optimal_accuracy,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1': optimal_f1
        }

        print(f"\n{model_name}:")
        print(f"  Optimal Threshold: {optimal_thresh:.3f} (default: 0.500)")
        print(f"  At optimal threshold:")
        print(f"    Recall: {optimal_recall:.4f} (was {model_result['recall']:.4f}) {'↑' if optimal_recall > model_result['recall'] else ''}")
        print(f"    Precision: {optimal_precision:.4f} (was {model_result['precision']:.4f})")
        print(f"    F1: {optimal_f1:.4f} (was {model_result['f1']:.4f})")
        print(f"    F2: {optimal_f2:.4f}")

        # Log to W&B
        if pipeline.use_wandb and pipeline.wandb_run:
            # Log optimal threshold metrics
            wandb.log({
                f"{model_name}/optimal_threshold": optimal_thresh,
                f"{model_name}/optimal/recall": optimal_recall,
                f"{model_name}/optimal/precision": optimal_precision,
                f"{model_name}/optimal/f1": optimal_f1,
                f"{model_name}/optimal/f2": optimal_f2
            })

            # Log optimal confusion matrix (使用 optimal threshold)
            y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
            y_pred_optimal_array = y_pred_optimal if isinstance(y_pred_optimal, np.ndarray) else np.array(y_pred_optimal)

            wandb.log({
                f"{model_name}/confusion_matrix_optimal_threshold": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_test_array.tolist(),
                    preds=y_pred_optimal_array.tolist(),
                    class_names=['Normal', 'Default'],
                    title=f"{model_name} Confusion Matrix (Optimal Threshold={optimal_thresh:.3f}, F2-optimized)"
                )
            })

    # 16. Stacking Ensemble
    print("\n" + "=" * 80)
    print("Stacking Ensemble Model")
    print("=" * 80)

    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    # Create base models list
    base_models = [
        ('xgb', results['XGBoost']['model']),
        ('lgb', results['LightGBM']['model']),
        ('cat', results['CatBoost']['model'])
    ]

    # Create stacking model with Logistic Regression as meta-learner
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
        'model': stacking_model,
        'auc': stack_auc,
        'accuracy': stack_accuracy,
        'precision': stack_precision,
        'recall': stack_recall,
        'f1': stack_f1,
        'predictions': stack_pred,
        'probabilities': stack_pred_proba
    }

    # Optimize Stacking threshold
    stack_optimal_thresh, stack_optimal_f2 = find_optimal_threshold(y_test, stack_pred_proba, metric='f2', beta=2)
    stack_pred_optimal = (stack_pred_proba >= stack_optimal_thresh).astype(int)
    stack_optimal_recall = recall_score(y_test, stack_pred_optimal)
    stack_optimal_precision = precision_score(y_test, stack_pred_optimal)
    stack_optimal_f1 = f1_score(y_test, stack_pred_optimal)

    print(f"\nStacking - Optimal Threshold: {stack_optimal_thresh:.3f}")
    print(f"  At optimal threshold:")
    print(f"    Recall: {stack_optimal_recall:.4f} (was {stack_recall:.4f}) {'↑' if stack_optimal_recall > stack_recall else ''}")
    print(f"    Precision: {stack_optimal_precision:.4f} (was {stack_precision:.4f})")
    print(f"    F1: {stack_optimal_f1:.4f} (was {stack_f1:.4f})")
    print(f"    F2: {stack_optimal_f2:.4f}")

    optimal_thresholds['Stacking'] = {
        'threshold': stack_optimal_thresh,
        'f2_score': stack_optimal_f2,
        'precision': stack_optimal_precision,
        'recall': stack_optimal_recall,
        'f1': stack_optimal_f1
    }

    # Log Stacking optimal confusion matrix to W&B
    if pipeline.use_wandb and pipeline.wandb_run:
        y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        stack_pred_optimal_array = stack_pred_optimal if isinstance(stack_pred_optimal, np.ndarray) else np.array(stack_pred_optimal)

        wandb.log({
            "Stacking/confusion_matrix_optimal": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test_array.tolist(),
                preds=stack_pred_optimal_array.tolist(),
                class_names=['Normal', 'Default']
            ),
            "Stacking/optimal_threshold": stack_optimal_thresh,
            "Stacking/optimal/recall": stack_optimal_recall,
            "Stacking/optimal/precision": stack_optimal_precision,
            "Stacking/optimal/f1": stack_optimal_f1,
            "Stacking/optimal/f2": stack_optimal_f2
        })

    # Update best model if stacking is better
    if stack_auc > best_auc:
        best_model_name = 'Stacking'
        best_auc = stack_auc
        print(f"\n[UPDATE] New best model: Stacking (AUC: {stack_auc:.4f})")

    # Log stacking to W&B
    if pipeline.use_wandb and pipeline.wandb_run:
        wandb.log({
            "Stacking/test/auc_roc": stack_auc,
            "Stacking/test/accuracy": stack_accuracy,
            "Stacking/test/precision": stack_precision,
            "Stacking/test/recall": stack_recall,
            "Stacking/test/f1": stack_f1,
            "best_auc_final": best_auc
        })

        # Create comprehensive model comparison visualization
        model_comparison_data = []
        for model_name, model_result in results.items():
            model_comparison_data.append([
                model_name,
                model_result['auc'],
                model_result['precision'],
                model_result['recall'],
                model_result['f1']
            ])

        model_comparison_table = wandb.Table(
            data=model_comparison_data,
            columns=["Model", "AUC", "Precision", "Recall", "F1"]
        )

        # Create multiple chart types for better visualization
        wandb.log({
            "model_comparison/metrics_table": model_comparison_table,
            "model_comparison/auc_bar_chart": wandb.plot.bar(
                model_comparison_table, "Model", "AUC",
                title=f"Model AUC Comparison (Best: {best_model_name} - {best_auc:.4f})"
            ),
            "model_comparison/recall_bar_chart": wandb.plot.bar(
                model_comparison_table, "Model", "Recall",
                title="Model Recall Comparison"
            ),
            "model_comparison/f1_bar_chart": wandb.plot.bar(
                model_comparison_table, "Model", "F1",
                title="Model F1-Score Comparison"
            ),
            "best_model/name": wandb.Html(f"<h2 style='color: green;'>Best Model: {best_model_name}</h2><p>AUC: {best_auc:.4f}</p>"),
            "best_model/auc_score": best_auc
        })

    # 16. 記錄資料集資訊到 W&B
    if pipeline.use_wandb and pipeline.wandb_run:
        dataset_info = {
            'dataset_info/total_samples': len(df_model),
            'dataset_info/train_samples': len(X_train_final),
            'dataset_info/test_samples': len(X_test_final),
            'dataset_info/num_features': X_train_final.shape[1],
            'dataset_info/num_numerical_features': len(numerical_features),
            'dataset_info/num_woe_features': len(woe_cols),
            'dataset_info/default_rate': y_train.mean()
        }
        pipeline.log_to_wandb(dataset_info)

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)

    # Finish W&B run
    if pipeline.use_wandb and pipeline.wandb_run:
        print(f"\n[OK] W&B 結果已上傳: {pipeline.wandb_run.url}")
        wandb.finish()

    return pipeline, results, woe_encoder


def update_config_model_params(best_params, config_path='../config.py'):
    """
    更新 config.py 中的 MODEL_PARAMS

    Args:
        best_params: 最佳超參數字典
        config_path: config.py 的路徑

    Returns:
        success: 是否更新成功
    """
    try:
        # 讀取 config.py
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 找到 MODEL_PARAMS 的開始和結束位置
        start_idx = None
        end_idx = None
        brace_count = 0

        for i, line in enumerate(lines):
            if 'MODEL_PARAMS = {' in line:
                start_idx = i
                brace_count = 1
                continue

            if start_idx is not None:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    end_idx = i
                    break

        if start_idx is None or end_idx is None:
            print("[ERROR] Could not find MODEL_PARAMS in config.py")
            return False

        # 建立新的 MODEL_PARAMS 內容
        metadata = best_params.get('metadata', {})
        new_params_lines = [
            "# Default model hyperparameters\n",
            f"# AUTO-UPDATED by WandB Sweep\n",
            f"# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"# Source: WandB run '{metadata.get('run_name', 'unknown')}'\n",
            f"# Best val_auc: {metadata.get('val_auc', 0):.4f}\n",
            "MODEL_PARAMS = {\n"
        ]

        # XGBoost 參數
        xgb_params = best_params.get('XGBoost', {})
        new_params_lines.append("    'XGBoost': {\n")
        for key, value in sorted(xgb_params.items()):
            if isinstance(value, str):
                new_params_lines.append(f"        '{key}': '{value}',\n")
            else:
                new_params_lines.append(f"        '{key}': {value},\n")
        # 補充必要的參數
        if 'random_state' not in xgb_params:
            new_params_lines.append("        'random_state': 42,\n")
        if 'eval_metric' not in xgb_params:
            new_params_lines.append("        'eval_metric': 'auc',\n")
        if 'verbosity' not in xgb_params:
            new_params_lines.append("        'verbosity': 0\n")
        else:
            # 移除最後一個逗號
            new_params_lines[-1] = new_params_lines[-1].rstrip(',\n') + '\n'
        new_params_lines.append("    },\n")

        # LightGBM 參數
        lgb_params = best_params.get('LightGBM', {})
        new_params_lines.append("    'LightGBM': {\n")
        for key, value in sorted(lgb_params.items()):
            if isinstance(value, str):
                new_params_lines.append(f"        '{key}': '{value}',\n")
            else:
                new_params_lines.append(f"        '{key}': {value},\n")
        # 補充必要的參數
        if 'is_unbalance' not in lgb_params:
            new_params_lines.append("        'is_unbalance': True,\n")
        if 'random_state' not in lgb_params:
            new_params_lines.append("        'random_state': 42,\n")
        if 'verbosity' not in lgb_params:
            new_params_lines.append("        'verbosity': -1\n")
        else:
            new_params_lines[-1] = new_params_lines[-1].rstrip(',\n') + '\n'
        new_params_lines.append("    },\n")

        # CatBoost 參數
        cat_params = best_params.get('CatBoost', {})
        new_params_lines.append("    'CatBoost': {\n")
        for key, value in sorted(cat_params.items()):
            if isinstance(value, str):
                new_params_lines.append(f"        '{key}': '{value}',\n")
            else:
                new_params_lines.append(f"        '{key}': {value},\n")
        # 補充必要的參數
        if 'auto_class_weights' not in cat_params:
            new_params_lines.append("        'auto_class_weights': 'Balanced',\n")
        if 'random_state' not in cat_params:
            new_params_lines.append("        'random_state': 42,\n")
        if 'verbose' not in cat_params:
            new_params_lines.append("        'verbose': False\n")
        else:
            new_params_lines[-1] = new_params_lines[-1].rstrip(',\n') + '\n'
        new_params_lines.append("    }\n")
        new_params_lines.append("}\n")

        # 替換 config.py 中的內容
        new_lines = lines[:start_idx] + new_params_lines + lines[end_idx+1:]

        # 寫回 config.py
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        print(f"\n[OK] config.py updated successfully!")
        print(f"     Path: {config_path}")
        print(f"     Source: {metadata.get('run_name', 'unknown')}")
        print(f"     Best val_auc: {metadata.get('val_auc', 0):.4f}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to update config.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_best_params_from_wandb(project_name="DPM-AutoTune", entity=None, update_config=True):
    """
    從 WandB 專案抓取最佳超參數並更新 config.py

    Args:
        project_name: WandB 專案名稱
        entity: WandB entity (用戶名)
        update_config: 是否自動更新 config.py

    Returns:
        best_params: 最佳超參數字典，如果失敗返回 None
    """
    try:
        import wandb
        api = wandb.Api()

        entity = entity or os.getenv('WANDB_ENTITY', 'yuczhen29-ccu')
        runs = api.runs(f"{entity}/{project_name}")

        # 過濾有 val_auc 的 runs 並抓取 val_auc 值 (避免 summary API 問題)
        valid_runs_with_auc = []
        print(f"[INFO] Scanning {len(list(runs))} runs...")

        for r in runs:
            try:
                # 嘗試多種方式讀取 summary
                summary_dict = None

                # 方法 1: _json_dict 屬性
                if hasattr(r.summary, '_json_dict'):
                    summary_dict = r.summary._json_dict
                # 方法 2: _summary 屬性
                elif hasattr(r, '_summary'):
                    summary_dict = r._summary
                # 方法 3: 嘗試轉換成字典
                elif hasattr(r.summary, '__iter__'):
                    try:
                        summary_dict = {k: v for k, v in r.summary.items()}
                    except:
                        pass

                if summary_dict and 'val_auc' in summary_dict:
                    val_auc = summary_dict['val_auc']
                    if isinstance(val_auc, (int, float)):
                        valid_runs_with_auc.append((r, float(val_auc)))
            except Exception as e:
                # 忽略無法讀取的 runs
                continue

        if not valid_runs_with_auc:
            print(f"[WARNING] No runs with val_auc found in {project_name}")
            return None

        print(f"[INFO] Found {len(valid_runs_with_auc)} runs with val_auc")

        # 找最佳 run
        best_run, best_auc_value = sorted(valid_runs_with_auc, key=lambda x: x[1], reverse=True)[0]

        print("\n" + "=" * 70)
        print("Best Run from WandB")
        print("=" * 70)
        print(f"Run Name: {best_run.name}")
        print(f"Run ID: {best_run.id}")
        print(f"val_auc: {best_auc_value:.4f}")
        print(f"URL: {best_run.url}")

        # 提取超參數
        best_params = {
            'XGBoost': {},
            'LightGBM': {},
            'CatBoost': {},
            'metadata': {
                'run_name': best_run.name,
                'run_id': best_run.id,
                'val_auc': best_auc_value,
                'updated_at': datetime.now().isoformat()
            }
        }

        # 確保 config 是字典格式
        config_dict = dict(best_run.config) if hasattr(best_run.config, 'items') else {}

        print(f"\n[DEBUG] Found {len(config_dict)} config parameters")

        # XGBoost 參數
        xgb_count = 0
        for key, value in config_dict.items():
            if key.startswith('xgb_'):
                param_name = key[4:]  # 移除 'xgb_' 前綴
                best_params['XGBoost'][param_name] = value
                xgb_count += 1
        print(f"  - XGBoost: {xgb_count} parameters")

        # LightGBM 參數
        lgb_count = 0
        for key, value in config_dict.items():
            if key.startswith('lgb_'):
                param_name = key[4:]  # 移除 'lgb_' 前綴
                best_params['LightGBM'][param_name] = value
                lgb_count += 1
        print(f"  - LightGBM: {lgb_count} parameters")

        # CatBoost 參數
        cat_count = 0
        for key, value in config_dict.items():
            if key.startswith('catboost_'):
                param_name = key[9:]  # 移除 'catboost_' 前綴
                best_params['CatBoost'][param_name] = value
                cat_count += 1
        print(f"  - CatBoost: {cat_count} parameters")

        # 特徵工程參數
        best_params['feature_engineering'] = {
            'use_target_encoding': config_dict.get('use_target_encoding', False),
            'use_geo_risk': config_dict.get('use_geo_risk', False),
            'use_smote': config_dict.get('use_smote', False),
            'scale_pos_weight': config_dict.get('scale_pos_weight', 3.82),
        }

        # 檢查是否有足夠的參數
        if xgb_count == 0 and lgb_count == 0 and cat_count == 0:
            print("\n[WARNING] No model parameters found in run config")
            print("[INFO] Available config keys:")
            for key in list(config_dict.keys())[:10]:  # 顯示前 10 個
                print(f"  - {key}")
            return None

        # 儲存到 JSON
        json_path = 'Train/best_params.json'
        os.makedirs('Train', exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Best parameters saved to {json_path}")

        # 自動更新 config.py
        if update_config:
            print("\n" + "=" * 70)
            print("Updating config.py...")
            print("=" * 70)
            update_success = update_config_model_params(best_params, '../config.py')

            if update_success:
                print("\n✅ config.py has been updated with best parameters!")
                print("   Both main.py and main_wandb.py will now use these parameters.")
            else:
                print("\n⚠️ Failed to update config.py automatically")
                print("   But parameters are saved in Train/best_params.json")

        return best_params

    except Exception as e:
        print(f"[ERROR] Failed to fetch best params from WandB: {e}")
        return None


def train_with_wandb_sweep():
    """
    使用 W&B Sweep 進行自動超參數調優
    """
    import wandb

    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # 貝葉斯優化
        'metric': {
            'name': 'val_auc',
            'goal': 'maximize'
        },
        'parameters': {
            # XGBoost parameters
            'xgb_n_estimators': {'values': [100, 200, 300, 500]},
            'xgb_max_depth': {'values': [3, 4, 5, 6, 7]},
            'xgb_learning_rate': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 0.3},
            'xgb_subsample': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
            'xgb_colsample_bytree': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
            'xgb_min_child_weight': {'values': [1, 3, 5, 7]},
            'xgb_gamma': {'distribution': 'uniform', 'min': 0, 'max': 0.5},
            'xgb_reg_alpha': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 10},
            'xgb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 10},

            # LightGBM parameters
            'lgb_n_estimators': {'values': [100, 200, 300, 500]},
            'lgb_max_depth': {'values': [3, 4, 5, 6, 7, -1]},
            'lgb_learning_rate': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 0.3},
            'lgb_num_leaves': {'values': [15, 31, 63, 127, 255]},
            'lgb_min_child_samples': {'values': [10, 20, 30, 50]},
            'lgb_subsample': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
            'lgb_colsample_bytree': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
            'lgb_reg_alpha': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 10},
            'lgb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 10},

            # CatBoost parameters
            'catboost_iterations': {'values': [100, 200, 300, 500]},
            'catboost_depth': {'values': [3, 4, 5, 6, 7, 8]},
            'catboost_learning_rate': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 0.3},
            'catboost_l2_leaf_reg': {'distribution': 'log_uniform_values', 'min': 1, 'max': 10},
            'catboost_border_count': {'values': [32, 64, 128, 254]},

            # Feature engineering
            'use_target_encoding': {'values': [True, False]},
            'use_geo_risk': {'values': [True, False]},
            'use_smote': {'values': [True, False]},
            'smote_sampling_strategy': {'distribution': 'uniform', 'min': 0.2, 'max': 0.5},
            'scale_pos_weight': {'distribution': 'uniform', 'min': 2, 'max': 6}  # M1+ definition (20.75% default rate)
        }
    }

    def train_sweep():
        """Single sweep training run"""
        run = wandb.init()
        config = wandb.config

        # Load and preprocess data (same as train_with_real_data)
        pipeline_temp = AdvancedDefaultPredictionPipeline(random_state=42, use_wandb=False)

        df = pd.read_excel('source/DPM_merged_cleaned.xlsx', engine='openpyxl')
        df = pipeline_temp.handle_missing_values(df)
        df = pipeline_temp.convert_data_types(df)
        df = pipeline_temp.define_default(df)
        df = pipeline_temp.credit_focused_feature_engineering(df)

        # Remove first row if it's header
        df = df[df.index != 0].copy()

        # === Tier 1: Overdue Pattern Features (CRITICAL for 0.9+ AUC!) ===
        print("\n[Sweep] Adding Tier 1 Overdue Pattern Features...")
        overdue_encoder = OverduePatternEncoder()
        df = overdue_encoder.create_overdue_pattern_features(df)

        # Use unified feature lists (includes Tier 1 features)
        from feature_engineering import get_feature_lists
        categorical_features, numerical_features, _ = get_feature_lists()

        # Filter features that exist
        categorical_features = [f for f in categorical_features if f in df.columns]
        numerical_features = [f for f in numerical_features if f in df.columns]

        X = df[categorical_features + numerical_features].copy()
        y = df['Default'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # WoE Encoding
        woe_encoder = WoEEncoder()
        X_train_fe = woe_encoder.fit_transform(X_train, y_train, categorical_features)
        X_test_fe = woe_encoder.transform(X_test, categorical_features)

        # Optional: Target Encoding
        if config.use_target_encoding:
            target_encoder = TargetEncoder(smoothing=1.0)
            X_train_fe = target_encoder.fit_transform(X_train_fe, y_train, categorical_features)
            X_test_fe = target_encoder.transform(X_test_fe, categorical_features)

        # Optional: Geographic Risk
        if config.use_geo_risk:
            geo_encoder = GeographicRiskEncoder(min_samples=10)
            X_train_fe = geo_encoder.fit_transform(X_train_fe, y_train)
            X_test_fe = geo_encoder.transform(X_test_fe)

        # Select features
        woe_cols = [f'{f}_WoE' for f in categorical_features]
        feature_cols = numerical_features + woe_cols

        if config.use_target_encoding:
            feature_cols += [f'{f}_target_enc' for f in categorical_features if f'{f}_target_enc' in X_train_fe.columns]
        if config.use_geo_risk:
            feature_cols += [c for c in ['res_risk_score', 'city_risk_score', 'address_stability_risk'] if c in X_train_fe.columns]

        X_train_final = X_train_fe[[f for f in feature_cols if f in X_train_fe.columns]].fillna(0)
        X_test_final = X_test_fe[[f for f in feature_cols if f in X_test_fe.columns]].fillna(0)

        # Optional: SMOTE
        if config.use_smote:
            smote = SMOTE(sampling_strategy=0.3, random_state=42)
            X_train_final, y_train = smote.fit_resample(X_train_final, y_train)

        # Train models
        xgb_model = xgb.XGBClassifier(
            n_estimators=config.xgb_n_estimators, max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate, subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree, min_child_weight=config.xgb_min_child_weight,
            scale_pos_weight=config.scale_pos_weight, random_state=42, eval_metric='auc'
        )
        xgb_model.fit(X_train_final, y_train)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=config.lgb_n_estimators, max_depth=config.lgb_max_depth,
            learning_rate=config.lgb_learning_rate, num_leaves=config.lgb_num_leaves,
            subsample=0.8, colsample_bytree=0.8, is_unbalance=True, random_state=42
        )
        lgb_model.fit(X_train_final, y_train)

        # Evaluate
        xgb_pred = xgb_model.predict_proba(X_test_final)[:, 1]
        lgb_pred = lgb_model.predict_proba(X_test_final)[:, 1]
        ensemble_pred = (xgb_pred + lgb_pred) / 2

        wandb.log({
            'val_auc': roc_auc_score(y_test, ensemble_pred),
            'xgb_auc': roc_auc_score(y_test, xgb_pred),
            'lgb_auc': roc_auc_score(y_test, lgb_pred),
            'num_features': len(X_train_final.columns)
        })

    # 用於追蹤最佳 run
    global_best_auc = 0
    global_best_config = None

    def train_sweep_with_tracking():
        """包裝的 train_sweep，追蹤最佳配置"""
        nonlocal global_best_auc, global_best_config

        # 執行原本的訓練
        train_sweep()

        # 取得當前 run 的結果
        current_run = wandb.run
        if current_run:
            current_auc = current_run.summary.get('val_auc', 0)
            if current_auc > global_best_auc:
                global_best_auc = current_auc
                global_best_config = dict(current_run.config)
                print(f"\n[NEW BEST] val_auc: {current_auc:.4f}")

                # 即時儲存最佳參數
                best_params = {
                    'XGBoost': {},
                    'LightGBM': {},
                    'CatBoost': {},
                    'metadata': {
                        'run_name': current_run.name,
                        'run_id': current_run.id,
                        'val_auc': current_auc,
                        'updated_at': datetime.now().isoformat()
                    }
                }

                # 提取參數
                for key, value in global_best_config.items():
                    if key.startswith('xgb_'):
                        best_params['XGBoost'][key[4:]] = value
                    elif key.startswith('lgb_'):
                        best_params['LightGBM'][key[4:]] = value
                    elif key.startswith('catboost_'):
                        best_params['CatBoost'][key[9:]] = value

                # 儲存到 JSON
                json_path = 'Train/best_params.json'
                os.makedirs('Train', exist_ok=True)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(best_params, f, indent=2, ensure_ascii=False)

                # 更新 config.py
                try:
                    update_config_model_params(best_params, '../config.py')
                except:
                    pass

    sweep_id = wandb.sweep(sweep_config, project='DPM-AutoTune')
    print(f'\nW&B Sweep ID: {sweep_id}')
    print('Starting hyperparameter optimization...\n')
    wandb.agent(sweep_id, train_sweep_with_tracking, count=30)

    print("\n" + "=" * 80)
    print("✅ Sweep Completed!")
    print("=" * 80)
    print(f'View results: https://wandb.ai/yuczhen29-ccu/DPM-AutoTune/sweeps/{sweep_id}')

    if global_best_auc > 0:
        print("\n" + "=" * 80)
        print("✅ Best Parameters Saved!")
        print("=" * 80)
        print(f"Best val_auc: {global_best_auc:.4f}")
        print(f"Parameters saved to: Train/best_params.json")
        print(f"config.py has been updated!")
        print(f"\nNext step: Run training with best params")
        print(f"  python main_wandb.py --use-best")
        print(f"  or")
        print(f"  python main.py  (也會使用最佳參數)")
    else:
        print("\n[WARNING] No runs completed successfully")
        print("Please check the sweep results on WandB")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='DPM Model Training with WandB')
    parser.add_argument('--sweep', action='store_true',
                       help='Run W&B Sweep for hyperparameter tuning (自動調優)')
    parser.add_argument('--use-best', action='store_true',
                       help='Use best parameters from WandB (使用最佳參數)')
    parser.add_argument('--fetch-best', action='store_true',
                       help='Fetch best parameters from WandB (手動抓取最佳參數)')

    args = parser.parse_args()

    if args.sweep:
        # 執行 Sweep 並自動抓取最佳參數
        print("=" * 80)
        print("Starting W&B Sweep for Hyperparameter Tuning")
        print("=" * 80)
        train_with_wandb_sweep()

    elif args.fetch_best:
        # 手動抓取最佳參數
        print("=" * 80)
        print("Fetching Best Parameters from WandB")
        print("=" * 80)

        best_params = get_best_params_from_wandb(project_name='DPM-AutoTune')

        if best_params:
            print(f"\n[SUCCESS] Best parameters saved to Train/best_params.json")
            print(f"\nNext step: Run training with best parameters:")
            print(f"  python main_wandb.py --use-best")
        else:
            print(f"\n[ERROR] Failed to fetch best parameters")

    else:
        # 正常訓練（可選擇是否使用最佳參數）
        pipeline, results, woe_encoder = train_with_real_data(
            use_best_params=args.use_best,
            best_params_path='Train/best_params.json'
        )