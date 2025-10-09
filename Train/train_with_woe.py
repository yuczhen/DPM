# -*- coding: utf-8 -*-
"""
DPM Model Training with WoE Features
結合東吳 WoE 特徵工程與 DPM 梯度提升模型
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# 匯入 WoE 編碼器
from woe_feature_engineering import WoEEncoder


def load_and_prepare_data():
    """載入並準備數據"""

    print("=" * 70)
    print("載入數據並定義目標變數")
    print("=" * 70)

    # 載入數據
    df = pd.read_excel('source/DPM_data_cleaned.xlsx', engine='openpyxl')

    # 定義違約
    def classify_default(overdue_status):
        status_str = str(overdue_status).strip()
        if status_str in ['Current', 'M0', 'M1']:
            return 0
        else:
            try:
                if status_str.startswith('M'):
                    m_number = int(status_str[1:])
                    return 1 if m_number >= 2 else 0
            except:
                pass
            return 0

    df['Default'] = df['overdue status'].apply(classify_default)

    print(f"總樣本數: {len(df):,}")
    print(f"違約樣本: {df['Default'].sum():,} ({df['Default'].mean():.2%})")
    print(f"正常樣本: {(df['Default']==0).sum():,} ({(df['Default']==0).mean():.2%})")

    return df


def prepare_features(df):
    """準備特徵"""

    # 分類特徵（用於 WoE）
    categorical_features = [
        'education',
        'residence status',
        'product',
        'marriage',
        'main business',
    ]

    # 數值特徵（可直接使用）
    numerical_features = [
        'loan term',
        'paid installments',
        'number of overdue before the first month',
        'number of overdue in the first half of the first month',
        'number of overdue in the second half of the first month',
    ]

    # 過濾實際存在的欄位
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]

    return categorical_features, numerical_features


def train_woe_enhanced_models(df):
    """訓練加入 WoE 特徵的模型"""

    print("\n" + "=" * 70)
    print("特徵準備")
    print("=" * 70)

    # 準備特徵
    categorical_features, numerical_features = prepare_features(df)

    print(f"\n分類特徵 ({len(categorical_features)}): {categorical_features}")
    print(f"數值特徵 ({len(numerical_features)}): {numerical_features}")

    # 處理數值特徵的缺失值和類型
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 準備 X 和 y
    all_features = categorical_features + numerical_features
    X = df[all_features].copy()
    y = df['Default']

    # 訓練-測試分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n訓練集: {len(X_train):,} 樣本")
    print(f"測試集: {len(X_test):,} 樣本")

    # ========== WoE 編碼 ==========
    print("\n" + "=" * 70)
    print("WoE 特徵工程")
    print("=" * 70)

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

    # 合併 WoE 特徵和原始數值特徵
    woe_cols = [col for col in X_train_woe.columns if '_WoE' in col]

    X_train_combined = pd.concat([
        X_train[numerical_features].reset_index(drop=True),
        X_train_woe[woe_cols].reset_index(drop=True)
    ], axis=1)

    X_test_combined = pd.concat([
        X_test[numerical_features].reset_index(drop=True),
        X_test_woe[woe_cols].reset_index(drop=True)
    ], axis=1)

    print(f"\n最終特徵數: {X_train_combined.shape[1]}")
    print(f"  - 數值特徵: {len(numerical_features)}")
    print(f"  - WoE 特徵: {len(woe_cols)}")

    # ========== 模型訓練 ==========
    print("\n" + "=" * 70)
    print("模型訓練與評估")
    print("=" * 70)

    models = {}
    results = {}

    # 1. XGBoost
    print("\n[1/3] 訓練 XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )
    xgb_model.fit(X_train_combined, y_train)
    models['XGBoost'] = xgb_model

    # 評估
    y_pred_proba = xgb_model.predict_proba(X_test_combined)[:, 1]
    y_pred = xgb_model.predict(X_test_combined)
    auc = roc_auc_score(y_test, y_pred_proba)
    results['XGBoost'] = {'AUC': auc, 'predictions': y_pred}

    print(f"  XGBoost AUC: {auc:.4f}")

    # 2. LightGBM
    print("\n[2/3] 訓練 LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    lgb_model.fit(X_train_combined, y_train)
    models['LightGBM'] = lgb_model

    y_pred_proba = lgb_model.predict_proba(X_test_combined)[:, 1]
    y_pred = lgb_model.predict(X_test_combined)
    auc = roc_auc_score(y_test, y_pred_proba)
    results['LightGBM'] = {'AUC': auc, 'predictions': y_pred}

    print(f"  LightGBM AUC: {auc:.4f}")

    # 3. CatBoost
    print("\n[3/3] 訓練 CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        random_state=42,
        auto_class_weights='Balanced',
        verbose=False
    )
    cat_model.fit(X_train_combined, y_train)
    models['CatBoost'] = cat_model

    y_pred_proba = cat_model.predict_proba(X_test_combined)[:, 1]
    y_pred = cat_model.predict(X_test_combined)
    auc = roc_auc_score(y_test, y_pred_proba)
    results['CatBoost'] = {'AUC': auc, 'predictions': y_pred}

    print(f"  CatBoost AUC: {auc:.4f}")

    # ========== 結果比較 ==========
    print("\n" + "=" * 70)
    print("模型性能比較")
    print("=" * 70)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  AUC-ROC: {result['AUC']:.4f}")
        print(f"\n{classification_report(y_test, result['predictions'], target_names=['Normal', 'Default'])}")

    # 找出最佳模型
    best_model = max(results.keys(), key=lambda x: results[x]['AUC'])
    best_auc = results[best_model]['AUC']

    print("\n" + "=" * 70)
    print(f"最佳模型: {best_model} (AUC = {best_auc:.4f})")
    print("=" * 70)

    return models, results, woe_encoder


if __name__ == "__main__":
    # 載入數據
    df = load_and_prepare_data()

    # 訓練模型
    models, results, woe_encoder = train_woe_enhanced_models(df)

    print("\n✓ 完成！WoE 增強的 DPM 模型訓練完成")
