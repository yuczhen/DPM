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

# 機器學習相關
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
from sklearn.linear_model import LogisticRegression

# SHAP解釋性
import shap

# 其他工具
import joblib
import json
from datetime import datetime
import os
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sys
sys.path.append('..')
import config

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

        # 初始化W&B
        if self.use_wandb:
            self._init_wandb()
        
    def generate_dataset(self, n_samples=10000, imbalance_ratio=0.15):
        """
        生成更真實的企業級資料集
        模擬銀行已放貸客戶的完整資料
        """
        np.random.seed(self.random_state)
        
        print(f"生成 {n_samples} 筆企業級訓練資料...")
        
        # === 基本人口統計學特徵 ===
        ages = np.random.normal(38, 12, n_samples).clip(18, 75)
        
        # 收入分佈 (對數正態分佈)
        log_incomes = np.random.normal(10.8, 0.7, n_samples)  # 約40K-100K主要區間
        incomes = np.exp(log_incomes).clip(20000, 500000)
        
        # 教育程度 (影響收入和違約率)
        education_levels = np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                         p=[0.15, 0.25, 0.35, 0.20, 0.05])  # 1:國中以下, 2:高中, 3:大專, 4:大學, 5:研究所
        
        # 職業類別
        job_categories = np.random.choice(range(1, 8), n_samples, 
                                        p=[0.20, 0.15, 0.18, 0.12, 0.15, 0.10, 0.10])
        # 1:服務業, 2:製造業, 3:金融, 4:科技, 5:教育, 6:醫療, 7:其他
        
        # === 貸款特徵 ===
        # 貸款金額 (與收入相關)
        loan_to_income_ratios = np.random.beta(2, 5, n_samples) * 3  # 0-3倍年收入
        loan_amounts = incomes * loan_to_income_ratios * np.random.uniform(0.1, 0.8, n_samples)
        loan_amounts = loan_amounts.clip(10000, 2000000)
        
        # 貸款期限
        loan_terms = np.random.choice([12, 18, 24, 30, 36, 48, 60], n_samples,
                                    p=[0.05, 0.10, 0.25, 0.15, 0.25, 0.15, 0.05])
        
        # 利率 (與風險相關)
        base_rates = np.random.normal(8, 2, n_samples).clip(4, 20)
        
        # === 信用歷史特徵 ===
        credit_history_months = np.random.exponential(36, n_samples).clip(0, 300)  # 信用歷史月數
        previous_loans_count = np.random.poisson(2.5, n_samples).clip(0, 15)  # 過往貸款次數
        
        # === 行為特徵 (放貸後6個月的表現) ===
        # 還款表現
        on_time_payment_rate = np.random.beta(8, 2, n_samples)  # 準時還款比率
        late_payments_count = np.random.poisson(1.2, n_samples).clip(0, 10)  # 遲繳次數
        missed_payments_count = np.random.poisson(0.3, n_samples).clip(0, 5)  # 漏繳次數
        
        # 賬戶活動
        avg_monthly_balance = incomes * np.random.uniform(0.1, 3.0, n_samples)  # 平均月餘額
        transaction_frequency = np.random.poisson(25, n_samples).clip(1, 100)  # 月交易次數
        
        # 信用卡使用情況
        credit_limit_total = loan_amounts * np.random.uniform(0.5, 5.0, n_samples)
        credit_utilization = np.random.beta(2, 3, n_samples)  # 信用使用率
        
        # === 財務健康指標 ===
        # 債務收入比
        total_monthly_debt = (loan_amounts / loan_terms) + np.random.uniform(500, 2000, n_samples)
        monthly_income = incomes / 12
        debt_to_income_ratio = total_monthly_debt / monthly_income
        
        # 資產相關
        has_mortgage = np.random.binomial(1, 0.4, n_samples)  # 是否有房貸
        has_car_loan = np.random.binomial(1, 0.3, n_samples)  # 是否有車貸
        savings_to_income_ratio = np.random.beta(1, 4, n_samples)  # 儲蓄收入比
        
        # === 外部數據特徵 ===
        # 就業狀況
        employment_status = np.random.choice([1, 2, 3, 4], n_samples, p=[0.75, 0.15, 0.05, 0.05])
        # 1:全職, 2:兼職, 3:自雇, 4:無業
        
        # 婚姻狀況
        marital_status = np.random.choice([1, 2, 3], n_samples, p=[0.45, 0.50, 0.05])
        # 1:單身, 2:已婚, 3:其他
        
        # 居住狀況
        housing_status = np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1])
        # 1:自有, 2:租賃, 3:其他
        
        # === 時間序列特徵 ===
        # 經濟環境指標 (模擬放貸時的經濟狀況)
        economic_index = np.random.normal(100, 15, n_samples).clip(70, 130)
        unemployment_rate = np.random.uniform(3, 8, n_samples)
        
        # 季節性因素
        loan_month = np.random.choice(range(1, 13), n_samples)
        is_holiday_season = np.where(np.isin(loan_month, [11, 12, 1]), 1, 0)
        
        # === 衍生特徵 ===
        # 風險分數計算
        risk_factors = (
            -education_levels * 0.1 +
            (employment_status == 4) * 0.8 +  # 無業
            late_payments_count * 0.15 +
            missed_payments_count * 0.3 +
            debt_to_income_ratio * 0.5 +
            (1 - on_time_payment_rate) * 0.4 +
            credit_utilization * 0.2 +
            (ages < 25) * 0.2 +  # 年輕風險
            (ages > 65) * 0.15 +  # 高齡風險
            np.random.normal(0, 0.3, n_samples)  # 隨機因子
        )
        
        # 違約機率計算 (sigmoid轉換)
        default_probability = 1 / (1 + np.exp(-risk_factors + 0.5))  # 調整基準線
        
        # 根據期望違約率調整
        target_default_rate = imbalance_ratio
        current_mean = np.mean(default_probability)
        adjustment = np.log(target_default_rate / (1 - target_default_rate)) - np.log(current_mean / (1 - current_mean))
        adjusted_probability = 1 / (1 + np.exp(-(risk_factors + adjustment)))
        
        # 生成最終違約標籤
        defaults = np.random.binomial(1, adjusted_probability, n_samples)
        
        # === 建立DataFrame ===
        data = pd.DataFrame({
            # 基本特徵
            'Customer_ID': range(1, n_samples + 1),
            'Age': ages.astype(int),
            'Education_Level': education_levels,
            'Job_Category': job_categories,
            'Employment_Status': employment_status,
            'Marital_Status': marital_status,
            'Housing_Status': housing_status,
            
            # 收入財務
            'Annual_Income': incomes.round(0).astype(int),
            'Monthly_Income': (incomes / 12).round(0).astype(int),
            
            # 貸款資訊
            'Loan_Amount': loan_amounts.round(0).astype(int),
            'Loan_Term_Months': loan_terms,
            'Interest_Rate': base_rates.round(2),
            'Loan_to_Income_Ratio': loan_to_income_ratios.round(3),
            
            # 信用歷史
            'Credit_History_Months': credit_history_months.round(0).astype(int),
            'Previous_Loans_Count': previous_loans_count,
            
            # 還款行為 (放貸後6個月觀察期)
            'On_Time_Payment_Rate': on_time_payment_rate.round(3),
            'Late_Payments_Count': late_payments_count,
            'Missed_Payments_Count': missed_payments_count,
            
            # 賬戶活動
            'Avg_Monthly_Balance': avg_monthly_balance.round(0).astype(int),
            'Transaction_Frequency': transaction_frequency,
            
            # 信用使用
            'Total_Credit_Limit': credit_limit_total.round(0).astype(int),
            'Credit_Utilization_Rate': credit_utilization.round(3),
            
            # 債務情況
            'Total_Monthly_Debt_Payment': total_monthly_debt.round(0).astype(int),
            'Debt_to_Income_Ratio': debt_to_income_ratio.round(3),
            'Has_Mortgage': has_mortgage,
            'Has_Car_Loan': has_car_loan,
            'Savings_to_Income_Ratio': savings_to_income_ratio.round(3),
            
            # 外部環境
            'Economic_Index_At_Origination': economic_index.round(1),
            'Unemployment_Rate_At_Origination': unemployment_rate.round(2),
            'Loan_Origination_Month': loan_month,
            'Is_Holiday_Season': is_holiday_season,
            
            # 目標變數
            'Default': defaults
        })
        
        print(f"資料集生成完成:")
        print(f"- 總樣本數: {len(data):,}")
        print(f"- 特徵數: {data.shape[1] - 2}")  # 排除ID和目標變數
        print(f"- 違約率: {data['Default'].mean():.2%}")
        print(f"- 違約樣本數: {data['Default'].sum():,}")
        print(f"- 正常樣本數: {(~data['Default'].astype(bool)).sum():,}")
        
        return data
    
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
    
    def create_preprocessing_pipeline(self, use_simple_imputer=False):
        """
        創建資料預處理Pipeline

        Args:
            use_simple_imputer (bool): 使用簡單填補(適合小資料集) vs KNN填補(適合大資料集)
        """
        # 數值特徵預處理
        if use_simple_imputer:
            # 簡單填補 - 適合小資料集 (< 10K 樣本)
            print("   使用 SimpleImputer (median) - 適合小資料集")
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # 使用中位數填補
                ('scaler', RobustScaler())  # 使用RobustScaler對異常值更穩健
            ])
        else:
            # KNN填補 - 適合大資料集 (適合捕捉特徵間關係)
            print("   使用 KNNImputer - 適合大資料集")
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
        if len(self.categorical_features) > 0:
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ]
            )
        else:
            # 如果沒有分類特徵，只處理數值特徵
            print("   [WARNING] 沒有分類特徵，只處理數值特徵")
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.numerical_features)
                ]
            )

        return self.preprocessing_pipeline

    def validate_data(self, data):
        """
        資料驗證：檢查資料品質問題

        Args:
            data (pd.DataFrame): 輸入資料

        Returns:
            pd.DataFrame: 清理後的資料
        """
        print("\n=== 資料驗證 ===")

        # 1. 檢查資料形狀
        print(f"原始資料形狀: {data.shape}")

        # 2. 檢查目標變數
        if 'Default' in data.columns:
            print(f"目標變數 'Default' 分佈:")
            print(data['Default'].value_counts())

            # 檢查目標變數類型
            if data['Default'].dtype == 'object' or data['Default'].dtype == 'string':
                print(f"[WARNING] 警告: 目標變數是文字類型，需要轉換為 0/1")
                # 顯示前幾個值作為參考
                print(f"   樣本值: {data['Default'].head(3).tolist()}")
            else:
                # 只有當是數值型態時才計算平均
                try:
                    print(f"違約率: {data['Default'].mean():.2%}")
                except:
                    print(f"[WARNING] 無法計算違約率")

                # 檢查目標變數是否為二元
                unique_values = data['Default'].unique()
                if len(unique_values) > 2:
                    print(f"[WARNING] 警告: 目標變數有 {len(unique_values)} 個不同值，應該只有 0/1")
        else:
            print("[WARNING] 警告: 找不到目標變數 'Default'")

        # 3. 檢查無限值
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            print(f"[WARNING] 發現無限值:")
            print(inf_counts[inf_counts > 0])
            # 替換無限值為 NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            print("   已將無限值替換為 NaN")

        # 4. 檢查缺失值
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"缺失值統計:")
            missing_df = pd.DataFrame({
                '欄位': missing_counts[missing_counts > 0].index,
                '缺失數': missing_counts[missing_counts > 0].values,
                '缺失率': (missing_counts[missing_counts > 0] / len(data) * 100).values
            })
            print(missing_df.to_string(index=False))
        else:
            print("[OK] 沒有缺失值")

        # 5. 檢查數值特徵的範圍
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n數值特徵統計:")
            for col in numerical_cols:
                if col != 'Default':
                    min_val = data[col].min()
                    max_val = data[col].max()
                    mean_val = data[col].mean()
                    print(f"  {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")

        # 6. 檢查是否有全為常數的欄位
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if len(constant_cols) > 0:
            print(f"\n[WARNING] 警告: 以下欄位只有單一值 (考慮移除):")
            for col in constant_cols:
                print(f"  - {col}")

        print("=" * 50)
        return data

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
        print(f"- 訓練集: {len(X_train):,} 樣本 (違約率: {y_train.mean():.2%}, 類別數: {y_train.nunique()})")
        print(f"- 驗證集: {len(X_val):,} 樣本 (違約率: {y_val.mean():.2%}, 類別數: {y_val.nunique()})")
        print(f"- 測試集: {len(X_test):,} 樣本 (違約率: {y_test.mean():.2%}, 類別數: {y_test.nunique()})")

        # 檢查每個集合是否有兩個類別
        if y_train.nunique() < 2:
            raise ValueError(
                f"[ERROR] 錯誤: 訓練集只有單一類別 ({y_train.unique()})，無法訓練模型！\n"
                f"   原因: 資料中沒有違約樣本，請檢查:\n"
                f"   1. status 欄位的值是否包含違約案件\n"
                f"   2. classify_status() 函數是否正確分類\n"
                f"   建議: 查看上方的 'status 分佈' 和 '各狀態的分類結果'"
            )
        if y_val.nunique() < 2:
            print("[WARNING] 警告: 驗證集只有單一類別，評估指標會不準確")
        if y_test.nunique() < 2:
            print("[WARNING] 警告: 測試集只有單一類別，評估指標會不準確")

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
    
    def initialize_models(self):
        """
        初始化多個模型，使用config中的預設參數
        """
        self.models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                solver='lbfgs',
                C=1.0,
                class_weight='balanced'  # 處理類別不平衡
            ),
            'XGBoost': xgb.XGBClassifier(
                **config.MODEL_PARAMS['XGBoost'],
                scale_pos_weight=2.0  # 處理類別不平衡
            ),
            'LightGBM': lgb.LGBMClassifier(
                **config.MODEL_PARAMS['LightGBM'],
                class_weight='balanced'  # 處理類別不平衡
            ),
            'CatBoost': CatBoostClassifier(
                **config.MODEL_PARAMS['CatBoost'],
                auto_class_weights='Balanced'  # 處理類別不平衡（注意大寫B）
            )
        }

        return self.models
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, cv_folds=5):
        """
        超參數調優
        """
        print("開始超參數調優...")
        
        # 定義搜索空間
        param_grids = {
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
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

            # Define custom x-axis for this model's training metrics
            if self.use_wandb and self.wandb_run:
                wandb.define_metric(f"{model_name}/epoch")
                wandb.define_metric(f"{model_name}/train/*", step_metric=f"{model_name}/epoch")

            # 訓練模型
            if model_name == 'XGBoost':
                # XGBoost 新版本使用不同的 early stopping 語法
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                # Log training history to W&B
                if self.use_wandb and self.wandb_run and hasattr(model, 'evals_result'):
                    eval_results = model.evals_result()
                    if 'validation_0' in eval_results:
                        for i, metric_val in enumerate(eval_results['validation_0'].get('logloss', [])):
                            wandb.log({
                                f"{model_name}/epoch": i,
                                f"{model_name}/train/logloss_epoch": metric_val
                            })
            elif model_name == 'LightGBM':
                # LightGBM 支持 early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                # Log training history to W&B
                if self.use_wandb and self.wandb_run and hasattr(model, 'evals_result_'):
                    eval_results = model.evals_result_
                    if 'valid_0' in eval_results:
                        for i in range(len(eval_results['valid_0'][list(eval_results['valid_0'].keys())[0]])):
                            log_dict = {f"{model_name}/epoch": i}
                            for metric_name, metric_vals in eval_results['valid_0'].items():
                                log_dict[f"{model_name}/train/{metric_name}_epoch"] = metric_vals[i]
                            wandb.log(log_dict)
            elif model_name == 'LogisticRegression':
                # Logistic Regression 不需要 early stopping
                model.fit(X_train, y_train)
            else:
                # CatBoost 等其他模型
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                # Log CatBoost training history
                if self.use_wandb and self.wandb_run and model_name == 'CatBoost' and hasattr(model, 'evals_result_'):
                    eval_results = model.evals_result_
                    if 'validation' in eval_results:
                        for i in range(len(eval_results['validation'][list(eval_results['validation'].keys())[0]])):
                            log_dict = {f"{model_name}/epoch": i}
                            for metric_name, metric_vals in eval_results['validation'].items():
                                log_dict[f"{model_name}/train/{metric_name}_epoch"] = metric_vals[i]
                            wandb.log(log_dict)
            
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

            # 打印結果
            self._print_model_results(model_name, results[model_name])

            # Log to W&B
            if self.use_wandb and self.wandb_run:
                # Log metrics for each dataset
                wandb_metrics = {}
                for dataset_name, metrics in results[model_name].items():
                    for metric_name, value in metrics.items():
                        wandb_metrics[f"{model_name}/{dataset_name}/{metric_name}"] = value

                self.log_to_wandb(wandb_metrics)

                # Log confusion matrix for test set
                if len(np.unique(y_test)) >= 2:
                    cm = confusion_matrix(y_test, test_pred)
                    # Convert to numpy arrays to avoid pandas index issues
                    y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
                    test_pred_array = test_pred if isinstance(test_pred, np.ndarray) else np.array(test_pred)

                    wandb.log({
                        f"{model_name}/confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=y_test_array.tolist(),
                            preds=test_pred_array.tolist(),
                            class_names=['Normal', 'Default']
                        )
                    })

        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """計算評估指標"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
        }

        # 檢查是否有兩個類別
        if len(np.unique(y_true)) >= 2:
            # 只有兩個類別時才計算這些指標
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba, labels=[0, 1])
        else:
            # 只有一個類別，設定為 N/A
            print(f"   [WARNING] 警告: y_true 只有單一類別 ({np.unique(y_true)}), 某些指標無法計算")
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
            metrics['auc_roc'] = 0.5  # 隨機猜測
            metrics['log_loss'] = float('inf')

        return metrics
    
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

        # 備份資料庫（如果有W&B記錄）
        if self.use_wandb and self.wandb_run:
            try:
                from db_logger import ExperimentDBLogger
                db_logger = ExperimentDBLogger()
                db_logger.backup_database()
                print("[OK] 資料庫備份完成")
            except Exception as e:
                print(f"[WARNING] 資料庫備份失敗: {e}")


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
            project_name = os.getenv('WANDB_PROJECT_NAME', 'dpm_credit_risk')
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
                'name': f"time_aware_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'config': {
                    'random_state': self.random_state,
                    'model_type': 'ensemble',
                    'data_version': os.getenv('DATA_VERSION', 'v2.0_time_aware'),
                    'split_method': 'temporal_split',
                    'use_recent_years': int(os.getenv('USE_RECENT_YEARS', 5)),
                },
                'tags': ['time_series', 'credit_risk', 'concept_drift', 'wandb_tracking'],
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

    def load_mini_dataset(self, data_path, n_samples=1000):
        """
        載入迷你資料集用於快速測試
        只選擇特定特徵：overdue days (F), paid installments features (AE~AH)

        Args:
            data_path (str): 資料檔案路徑
            n_samples (int): 樣本數量（從最新序號開始）

        Returns:
            pd.DataFrame: 迷你資料集
        """
        print(f"載入迷你資料集 (最新 {n_samples} 筆)...")

        # 讀取資料 (支援 CSV 和 Excel)
        if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            data = pd.read_excel(data_path)
        else:
            data = pd.read_csv(data_path, encoding='utf-8', low_memory=False)

        print(f"- 原始資料筆數: {len(data):,}")

        # 使用隨機抽樣而非最新資料（因為最新資料可能都是正常案件）
        print(f"   使用隨機抽樣（包含新舊案件）...")
        if len(data) > n_samples:
            data_mini = data.sample(n=n_samples, random_state=42).copy()
        else:
            data_mini = data.copy()
            print(f"   [WARNING] 資料總數 ({len(data)}) 少於需求 ({n_samples})，使用全部資料")

        # 選擇特定欄位
        # 注意：暫時移除 'overdue days'，因為它與 Default 高度相關（可能造成數據洩漏）
        # AE~AH columns: paid installments related features
        # status column: 用於創建目標變數 (Current/M1/M2... → Default)
        selected_columns = [
            # 'overdue days',  # 暫時移除：與目標變數高度相關
            'paid installments',  # AE column
            'number of overdue before the first month',  # AF column
            'number of overdue in the first half of the first month',  # AG column
            'number of overdue in the second half of the first month',  # AH column
            'status',  # 用於創建 Default 目標變數
        ]

        # 設定目標變數為 status（將用來創建 Default）
        target_col = 'status'
        print(f"\n   使用 'status' 欄位創建目標變數 (Current/M1/M2 → 0/1)")

        # 檢查欄位是否存在
        available_cols = [col for col in selected_columns if col in data_mini.columns]
        missing_cols = [col for col in selected_columns if col not in data_mini.columns]

        if missing_cols:
            print(f"[WARNING] 缺少欄位: {missing_cols}")

        data_mini_selected = data_mini[available_cols].copy()

        # 處理缺失值
        print(f"\n   處理缺失值前樣本數: {len(data_mini_selected):,}")
        print(f"   缺失值統計:")
        print(data_mini_selected.isnull().sum())

        data_mini_selected = data_mini_selected.dropna()

        print(f"   處理缺失值後樣本數: {len(data_mini_selected):,}")

        print(f"\n[OK] 迷你資料集載入完成:")
        print(f"- 樣本數: {len(data_mini_selected):,}")
        print(f"- 特徵數: {len(available_cols) - (1 if target_col else 0)}")
        print(f"- 特徵列表: {[col for col in available_cols if col != target_col]}")

        if target_col and target_col in data_mini_selected.columns:
            print(f"- 目標變數: {target_col}")
            print(f"- 目標變數分佈 (原始):")
            print(data_mini_selected[target_col].value_counts().head(10))

            # 轉換目標變數為 0/1 二元值
            print(f"\n[DEBUG] 開始轉換目標變數，target_col = '{target_col}'")
            if target_col == 'default rate':
                # 使用 default rate 來定義違約
                # 如果 default rate 是百分比字串，需要轉換
                print(f"   使用 default rate 建立二元目標變數...")

                # 先檢查並轉換為數值
                print(f"   default rate 資料型態: {data_mini_selected[target_col].dtype}")

                if data_mini_selected[target_col].dtype == 'object':
                    # 移除百分比符號並轉換
                    data_mini_selected['default_rate_numeric'] = data_mini_selected[target_col].astype(str).str.replace('%', '').astype(float)
                else:
                    # 已經是數值，直接使用
                    data_mini_selected['default_rate_numeric'] = pd.to_numeric(data_mini_selected[target_col], errors='coerce')

                # 移除 NaN 值
                data_mini_selected = data_mini_selected.dropna(subset=['default_rate_numeric'])

                print(f"   default rate 範圍: {data_mini_selected['default_rate_numeric'].min():.2f} ~ {data_mini_selected['default_rate_numeric'].max():.2f}")

                # 判斷 default rate 是百分比 (0-100) 還是比率 (0-1)
                max_rate = data_mini_selected['default_rate_numeric'].max()
                if max_rate <= 1.0:
                    # 比率格式 (0-1)，轉換為百分比
                    data_mini_selected['default_rate_numeric'] = data_mini_selected['default_rate_numeric'] * 100
                    print(f"   偵測到比率格式 (0-1)，已轉換為百分比")
                    print(f"   轉換後範圍: {data_mini_selected['default_rate_numeric'].min():.2f}% ~ {data_mini_selected['default_rate_numeric'].max():.2f}%")

                # 使用分位數來確保類別平衡
                # 目標: 讓 30% 的樣本為 Default=1
                target_positive_ratio = 0.3
                default_threshold = data_mini_selected['default_rate_numeric'].quantile(1 - target_positive_ratio)

                print(f"   default rate 統計:")
                print(f"   - 最小值: {data_mini_selected['default_rate_numeric'].min():.2f}%")
                print(f"   - 25th percentile: {data_mini_selected['default_rate_numeric'].quantile(0.25):.2f}%")
                print(f"   - 中位數: {data_mini_selected['default_rate_numeric'].median():.2f}%")
                print(f"   - 75th percentile: {data_mini_selected['default_rate_numeric'].quantile(0.75):.2f}%")
                print(f"   - 最大值: {data_mini_selected['default_rate_numeric'].max():.2f}%")

                data_mini_selected['Default'] = (data_mini_selected['default_rate_numeric'] >= default_threshold).astype(int)

                print(f"\n   使用 70th percentile 作為閾值: {default_threshold:.2f}%")
                print(f"   轉換後目標變數分佈:")
                default_counts = data_mini_selected['Default'].value_counts()
                print(default_counts)
                print(f"   違約率: {data_mini_selected['Default'].mean():.2%}")

                # 檢查是否有足夠的兩個類別
                if data_mini_selected['Default'].nunique() < 2:
                    print(f"\n   [WARNING] 警告: 只有一個類別，使用中位數作為閾值...")
                    default_threshold = data_mini_selected['default_rate_numeric'].median()
                    data_mini_selected['Default'] = (data_mini_selected['default_rate_numeric'] >= default_threshold).astype(int)
                    print(f"   調整後閾值: {default_threshold:.2f}%")
                    print(f"   最終目標變數分佈:")
                    print(data_mini_selected['Default'].value_counts())

                # 確保每個類別至少有 10 個樣本
                min_samples_per_class = 10
                class_counts = data_mini_selected['Default'].value_counts()
                if len(class_counts) < 2 or class_counts.min() < min_samples_per_class:
                    print(f"\n   [WARNING] 警告: 某個類別樣本數太少 (< {min_samples_per_class})")
                    print(f"   建議: 增加樣本數或調整閾值")

                # 刪除臨時欄位和原始 default rate
                columns_to_drop = ['default_rate_numeric']
                if target_col in data_mini_selected.columns and target_col != 'Default':
                    columns_to_drop.append(target_col)
                data_mini_selected.drop(columns=columns_to_drop, inplace=True)

                # 確認 Default 欄位存在
                if 'Default' not in data_mini_selected.columns:
                    print("   [WARNING] 錯誤: Default 欄位創建失敗！")
                else:
                    print(f"   [OK] Default 欄位已創建")

            elif data_mini_selected[target_col].dtype == 'object':
                print(f"   轉換 status 欄位為二元目標變數...")

                # 查看所有唯一值
                unique_statuses = data_mini_selected[target_col].unique()
                print(f"   發現的狀態類別數量: {len(unique_statuses)}")
                print(f"   所有狀態類別:")
                for i, status in enumerate(unique_statuses):
                    count = (data_mini_selected[target_col] == status).sum()
                    print(f"      {i+1}. '{status}' - {count} 筆")

                # 定義違約規則: 根據業務邏輯分類
                # M1~M2 = 正常早期結清
                # M3+ = 高風險/違約 (拖太久)
                def classify_status(status_text):
                    status_str = str(status_text).strip()

                    # === 高風險/違約狀態 (Class 1) ===
                    # 優先檢查，因為這些是明確的違約
                    default_keywords = [
                        '債權轉讓',      # 已轉給催收公司
                        '可能轉銷',      # 可能成為呆帳
                        '呆帳',          # 呆帳
                        '催收',          # 進入催收程序
                        '轉銷',          # 轉銷
                        '沖銷',          # 包含「逾期沖銷」
                        '違約',
                        '壞帳',
                        '損失',
                        'M3', 'M4', 'M5', 'M6'  # 拖超過 2 個月 = 高風險
                    ]

                    for keyword in default_keywords:
                        if keyword in status_str:
                            return 1

                    # === 正常狀態 (Class 0) ===
                    normal_keywords = [
                        '結清',          # 正常還清
                        '一年結清',      # 一年內還清
                        '二年結清',
                        '三年結清',
                        'M1',           # 第 1 個月結清 - 正常
                        'M2',           # 第 2 個月結清 - 可接受
                        'CURRENT',
                        '正常'
                    ]

                    for keyword in normal_keywords:
                        if keyword in status_str:
                            return 0

                    # === 排除的狀態 (返回 None，稍後過濾) ===
                    exclude_keywords = [
                        '已撥款',        # 結果未知，不能用於訓練
                        '未上傳',        # 非貸款案件
                        '作廢',          # 無效案件
                        '展延'           # 狀態不明確
                    ]

                    for keyword in exclude_keywords:
                        if keyword in status_str:
                            return None  # 標記為需排除

                    # 未知狀態也排除
                    return None

                # 測試分類函數
                print(f"\n   [DEBUG] 測試分類函數:")
                test_statuses = ['已撥款', '結清', '債權轉讓-一般催收', '呆帳轉銷', '可能轉銷']
                for ts in test_statuses:
                    if ts in data_mini_selected[target_col].values:
                        result = classify_status(ts)
                        print(f"      '{ts}' → {result}")

                data_mini_selected['Default'] = data_mini_selected[target_col].apply(classify_status)

                # 過濾掉 None 值（排除的案件）
                before_filter = len(data_mini_selected)
                data_mini_selected = data_mini_selected[data_mini_selected['Default'].notna()].copy()
                after_filter = len(data_mini_selected)

                print(f"\n   過濾不可用案件:")
                print(f"   - 原始樣本數: {before_filter}")
                print(f"   - 排除樣本數: {before_filter - after_filter} (已撥款/未上傳/作廢/展延等)")
                print(f"   - 可用樣本數: {after_filter}")

                # 轉換為整數
                data_mini_selected['Default'] = data_mini_selected['Default'].astype(int)

                print(f"\n   轉換後目標變數分佈:")
                print(data_mini_selected['Default'].value_counts())
                print(f"   違約率: {data_mini_selected['Default'].mean():.2%}")

                # 顯示每個 status 對應的 Default 值
                print(f"\n   各狀態的分類結果:")
                status_default_map = data_mini_selected.groupby(target_col)['Default'].agg(['first', 'count'])
                print(status_default_map)

                # 檢查是否至少有兩個類別
                if data_mini_selected['Default'].nunique() < 2:
                    print(f"   [WARNING] 警告: 只有單一類別，嘗試增加樣本數或調整分類規則")
                    print(f"   建議: 檢查原始 status 欄位的實際值並調整關鍵字")

                # 刪除原始 status 欄位
                if target_col != 'Default':
                    data_mini_selected.drop(columns=[target_col], inplace=True)
            else:
                # 數值型態，直接重命名
                if target_col != 'Default':
                    data_mini_selected.rename(columns={target_col: 'Default'}, inplace=True)

        # 記錄到W&B
        if self.use_wandb:
            self.log_to_wandb({
                'dataset_info/mini_dataset': True,
                'dataset_info/samples': len(data_mini_selected),
                'dataset_info/features': len(available_cols) - 1
            })

        # 最終檢查
        print(f"\n[DEBUG] load_mini_dataset 返回前:")
        print(f"  欄位: {data_mini_selected.columns.tolist()}")
        print(f"  是否有 Default: {'Default' in data_mini_selected.columns}")

        return data_mini_selected

    def simple_feature_engineering(self, data):
        """
        輕量級特徵工程 - 適用於迷你資料集
        只使用現有的欄位創建衍生特徵
        """
        print("\n進行輕量級特徵工程...")

        data_fe = data.copy()

        # 檢查是否有 Default 欄位
        if 'Default' not in data_fe.columns:
            raise ValueError("[WARNING] 錯誤: 資料中缺少 'Default' 欄位，請先執行 load_mini_dataset()")

        print(f"   輸入資料形狀: {data_fe.shape}")
        print(f"   包含 Default 欄位: {'Default' in data_fe.columns}")

        # 保留原始欄位名稱以便操作
        paid_inst = 'paid installments'
        overdue_before = 'number of overdue before the first month'
        overdue_1st_half = 'number of overdue in the first half of the first month'
        overdue_2nd_half = 'number of overdue in the second half of the first month'

        # 確保所有欄位都是數值型態（但不包括 Default）
        for col in [paid_inst, overdue_before, overdue_1st_half, overdue_2nd_half]:
            if col in data_fe.columns:
                data_fe[col] = pd.to_numeric(data_fe[col], errors='coerce').fillna(0)

        # 2. 總逾期次數
        data_fe['total_overdue_count'] = (
            data_fe[overdue_before] +
            data_fe[overdue_1st_half] +
            data_fe[overdue_2nd_half]
        )

        # 3. 早期逾期比率（第一個月的逾期占總逾期的比例）
        total_overdue = data_fe['total_overdue_count']
        data_fe['early_overdue_ratio'] = (
            (data_fe[overdue_1st_half] + data_fe[overdue_2nd_half]) / (total_overdue + 1)
        )

        # 4. 逾期惡化趨勢（第二半月比第一半月多）
        data_fe['overdue_worsening'] = (
            data_fe[overdue_2nd_half] > data_fe[overdue_1st_half]
        ).astype(int)

        # 5. 是否有任何逾期記錄
        data_fe['has_any_overdue'] = (data_fe['total_overdue_count'] > 0).astype(int)

        # 6. 繳款進度（假設總期數較多表示較穩定）
        data_fe['payment_progress_score'] = np.log1p(data_fe[paid_inst].astype(float))

        print(f"[OK] 特徵工程完成:")
        print(f"   - 原始特徵: 4")
        print(f"   - 衍生特徵: 5")
        print(f"   - 總特徵數: {data_fe.shape[1] - 1}")  # 扣除 Default

        return data_fe

    def load_and_filter_recent_data(self, data_path, years=5):
        """
        載入資料並篩選最近N年的資料

        Args:
            data_path (str): 資料檔案路徑
            years (int): 年數

        Returns:
            pd.DataFrame: 篩選後的資料
        """
        print(f"載入資料並篩選最近{years}年...")

        # 讀取資料 (支援 CSV 和 Excel)
        if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            data = pd.read_excel(data_path)
        else:
            data = pd.read_csv(data_path, encoding='utf-8', low_memory=False)

        # 轉換日期欄位
        data['進件日期'] = pd.to_datetime(data['進件日期'], errors='coerce')

        # 移除無效日期記錄
        data_valid = data.dropna(subset=['進件日期'])

        # 計算最近N年的起始日期
        recent_start_date = data_valid['進件日期'].max() - pd.DateOffset(years=years)

        # 篩選資料
        filtered_data = data_valid[data_valid['進件日期'] >= recent_start_date].copy()

        print("資料篩選完成:")
        print(f"- 原始資料筆數: {len(data_valid):,}")
        print(f"- 篩選後資料筆數: {len(filtered_data):,}")
        print(f"- 時間範圍: {recent_start_date.date()} 至 {data_valid['進件日期'].max().date()}")

        # 記錄到W&B
        if self.use_wandb:
            self.log_to_wandb({
                'dataset_info/original_samples': len(data_valid),
                'dataset_info/filtered_samples': len(filtered_data),
                'dataset_info/filtered_years': years,
                'dataset_info/start_date': str(recent_start_date.date()),
                'dataset_info/end_date': str(data_valid['進件日期'].max().date())
            })

        return filtered_data

    def temporal_train_test_split(self, data, train_ratio=0.7, val_ratio=0.15):
        """
        時間意識的訓練/驗證/測試集分割
        確保訓練集總是在測試集之前

        Args:
            data (pd.DataFrame): 輸入資料
            train_ratio (float): 訓練集比例
            val_ratio (float): 驗證集比例

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("執行時間意識的資料分割...")

        # 檢查時間欄位
        time_col = '進件日期'
        if time_col not in data.columns:
            print(f"[WARNING] 警告: 找不到時間欄位 '{time_col}'，使用隨機分割")
            return self.advanced_train_test_split(data, test_size=1-val_ratio, val_size=val_ratio)

        # 按時間排序
        data_sorted = data.sort_values(time_col).reset_index(drop=True)

        # 計算分割點
        n_total = len(data_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # 分割資料
        train_data = data_sorted.iloc[:n_train]
        val_data = data_sorted.iloc[n_train:n_train + n_val]
        test_data = data_sorted.iloc[n_train + n_val:]

        # 準備特徵和目標變數
        exclude_cols = ['Customer_ID', 'Default']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        X_train = train_data[feature_cols]
        y_train = train_data['Default']

        X_val = val_data[feature_cols]
        y_val = val_data['Default']

        X_test = test_data[feature_cols]
        y_test = test_data['Default']

        # 輸出分割資訊
        print("=== 時間意識資料分割完成 ===")
        print(f"總資料筆數: {n_total:,}")
        print(f"訓練集: {len(train_data):,} ({len(train_data)/n_total*100:.1f}%)")
        print(f"驗證集: {len(val_data):,} ({len(val_data)/n_total*100:.1f}%)")
        print(f"測試集: {len(test_data):,} ({len(test_data)/n_total*100:.1f}%)")

        print("\n時間範圍:")
        print(f"訓練集: {train_data[time_col].min().date()} 至 {train_data[time_col].max().date()}")
        print(f"驗證集: {val_data[time_col].min().date()} 至 {val_data[time_col].max().date()}")
        print(f"測試集: {test_data[time_col].min().date()} 至 {test_data[time_col].max().date()}")

        # 計算各集合的違約率
        for name, dataset, y in [('訓練集', train_data, y_train),
                                ('驗證集', val_data, y_val),
                                ('測試集', test_data, y_test)]:
            default_rate = y.mean() * 100
            print(f"{name}違約率: {default_rate:.2f}%")

        # 記錄到W&B
        if self.use_wandb:
            split_metrics = {
                'split/train_samples': len(train_data),
                'split/val_samples': len(val_data),
                'split/test_samples': len(test_data),
                'split/train_default_rate': y_train.mean(),
                'split/val_default_rate': y_val.mean(),
                'split/test_default_rate': y_test.mean(),
                'split/train_date_start': str(train_data[time_col].min().date()),
                'split/train_date_end': str(train_data[time_col].max().date()),
                'split/val_date_start': str(val_data[time_col].min().date()),
                'split/val_date_end': str(val_data[time_col].max().date()),
                'split/test_date_start': str(test_data[time_col].min().date()),
                'split/test_date_end': str(test_data[time_col].max().date())
            }
            self.log_to_wandb(split_metrics)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def advanced_train_test_split(self, data, test_size=0.2, val_size=0.1):
        """
        進階的訓練/驗證/測試集分割
        包含時間序列考量和分層抽樣
        """
        print("執行進階資料分割...")

        # 檢查是否應該使用時間意識分割
        if '進件日期' in data.columns:
            print("[OK] 偵測到時間欄位，使用時間意識分割")
            return self.temporal_train_test_split(data, 1-test_size-val_size, val_size)
        else:
            print("[WARNING] 未偵測到時間欄位，使用隨機分割")

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

        # 記錄到W&B
        if self.use_wandb:
            self.log_to_wandb({
                'split/train_samples': len(X_train),
                'split/val_samples': len(X_val),
                'split/test_samples': len(X_test),
                'split/train_default_rate': y_train.mean(),
                'split/val_default_rate': y_val.mean(),
                'split/test_default_rate': y_test.mean(),
                'split/method': 'random_split'
            })

        return X_train, X_val, X_test, y_train, y_val, y_test

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    快速訓練迷你資料集以測試模型性能
    """
    print("=" * 80)
    print("DPM 迷你資料集快速訓練")
    print("=" * 80)

    # 1. 初始化管道
    print("\n1. 初始化管道...")
    pipeline = AdvancedDefaultPredictionPipeline(
        random_state=42,
        use_wandb=True  # 啟用 W&B 追蹤實驗
    )

    # 初始化 W&B
    pipeline._init_wandb()

    # 2. 載入迷你資料集
    print("\n2. 載入迷你資料集...")
    import os
    # 確保路徑正確（從 Train 目錄或從根目錄執行都能找到）
    if os.path.exists('Source/NCCU_CRM_cleaned.xlsx'):
        data_path = 'Source/NCCU_CRM_cleaned.xlsx'
    else:
        data_path = 'Train/Source/NCCU_CRM_cleaned.xlsx'
    # 增加樣本數到 50000，確保包含足夠的違約案件
    mini_data = pipeline.load_mini_dataset(data_path, n_samples=50000)

    # DEBUG: 檢查載入後的資料
    print("\n[DEBUG] 載入後的資料欄位:")
    print(mini_data.columns.tolist())
    print(f"[DEBUG] 資料形狀: {mini_data.shape}")
    print(f"[DEBUG] 是否有 Default 欄位: {'Default' in mini_data.columns}")

    # 2.5. 資料驗證
    print("\n2.5 資料驗證...")
    mini_data = pipeline.validate_data(mini_data)

    # DEBUG: 檢查驗證後的資料
    print("\n[DEBUG] 驗證後的資料欄位:")
    print(mini_data.columns.tolist())
    print(f"[DEBUG] 是否有 Default 欄位: {'Default' in mini_data.columns}")

    # 2.6. 特徵工程
    print("\n2.6 特徵工程...")
    mini_data = pipeline.simple_feature_engineering(mini_data)

    # 3. 資料分割
    print("\n3. 資料分割...")
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.advanced_train_test_split(
        mini_data,
        test_size=0.2,
        val_size=0.1
    )

    # 4. 更新特徵列表（迷你資料集的特徵）
    print("\n4. 更新特徵列表...")
    pipeline.numerical_features = [col for col in X_train.columns]
    pipeline.categorical_features = []
    print(f"   數值特徵 ({len(pipeline.numerical_features)}): {pipeline.numerical_features}")
    print(f"   分類特徵 ({len(pipeline.categorical_features)}): {pipeline.categorical_features}")

    # 5. 建立預處理管道
    print("\n5. 建立預處理管道...")
    preprocessing = pipeline.create_preprocessing_pipeline(use_simple_imputer=True)  # 小資料集使用簡單填補

    # 6. 轉換資料
    print("\n6. 轉換資料...")
    print(f"   訓練集形狀 (轉換前): {X_train.shape}")
    X_train_processed = preprocessing.fit_transform(X_train)
    print(f"   訓練集形狀 (轉換後): {X_train_processed.shape}")

    X_val_processed = preprocessing.transform(X_val)
    print(f"   驗證集形狀 (轉換後): {X_val_processed.shape}")

    X_test_processed = preprocessing.transform(X_test)
    print(f"   測試集形狀 (轉換後): {X_test_processed.shape}")

    # 7. 初始化模型
    print("\n6. 初始化模型...")
    models = pipeline.initialize_models()
    print(f"   模型列表: {list(models.keys())}")

    # 8. 訓練和評估模型
    print("\n7. 訓練和評估模型...")
    results = pipeline.train_and_evaluate_models(
        X_train_processed, y_train,
        X_val_processed, y_val,
        X_test_processed, y_test
    )

    # 9. 顯示結果摘要
    print("\n" + "=" * 80)
    print("模型性能摘要 (測試集)")
    print("=" * 80)

    # 先顯示測試集的實際類別分佈
    print(f"\n測試集真實標籤分佈:")
    print(f"  - Class 0 (正常): {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")
    print(f"  - Class 1 (違約): {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")
    print(f"  - 總樣本數: {len(y_test)}")

    for model_name, metrics in results.items():
        test_metrics = metrics['test']
        print(f"\n{model_name}:")
        print(f"  - Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  - AUC-ROC:   {test_metrics['auc_roc']:.4f}")
        print(f"  - Precision: {test_metrics['precision']:.4f}")
        print(f"  - Recall:    {test_metrics['recall']:.4f}")
        print(f"  - F1-Score:  {test_metrics['f1']:.4f}")

    # 10. 找出最佳模型
    best_model = max(results.keys(), key=lambda x: results[x]['test']['auc_roc'])
    best_auc = results[best_model]['test']['auc_roc']

    print("\n" + "=" * 80)
    print(f"[BEST] 最佳模型: {best_model} (AUC-ROC: {best_auc:.4f})")
    print("=" * 80)

    # Log best model to W&B
    if pipeline.use_wandb and pipeline.wandb_run:
        wandb.log({
            "best_model": best_model,
            "best_auc_roc": best_auc
        })
        print(f"\n[OK] W&B 結果已上傳: {pipeline.wandb_run.url}")
        wandb.finish()

    print("\n[OK] 訓練完成！")
