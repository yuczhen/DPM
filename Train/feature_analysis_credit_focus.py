# -*- coding: utf-8 -*-
"""
Feature Analysis Based on Credit Assessment Principles
根據徵審人員關注點分析特徵重要性
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from woe_feature_engineering import WoEEncoder


def load_data():
    """載入數據"""
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
    return df


def create_derived_features(df):
    """
    創建衍生特徵
    根據徵審關注點：還款能力 + 聯絡穩定性
    """

    print("=" * 70)
    print("創建衍生特徵")
    print("=" * 70)

    df_new = df.copy()

    # ========== 財務能力指標 ==========
    print("\n【財務能力指標】")

    # 1. 還款進度比率
    df_new['loan term'] = pd.to_numeric(df_new['loan term'], errors='coerce')
    df_new['paid installments'] = pd.to_numeric(df_new['paid installments'], errors='coerce')

    df_new['payment_progress_ratio'] = (
        df_new['paid installments'] / df_new['loan term']
    ).fillna(0).clip(0, 1)

    print(f"  ✓ 還款進度比率 (已繳期數/總期數)")
    print(f"    範圍: {df_new['payment_progress_ratio'].min():.2f} ~ {df_new['payment_progress_ratio'].max():.2f}")

    # 2. 月薪等級分組
    df_new['month salary'] = pd.to_numeric(df_new['month salary'], errors='coerce')

    # 處理異常值 (月薪 < 1000 視為缺失)
    df_new.loc[df_new['month salary'] < 1000, 'month salary'] = np.nan

    valid_salary = df_new['month salary'].notna()
    if valid_salary.sum() > 0:
        df_new.loc[valid_salary, 'salary_level'] = pd.qcut(
            df_new.loc[valid_salary, 'month salary'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )
        print(f"  ✓ 月薪等級分組 (5個等級)")
        print(f"    有效樣本: {valid_salary.sum():,} ({valid_salary.sum()/len(df)*100:.1f}%)")

    # 3. 年資穩定性
    df_new['job tenure'] = pd.to_numeric(df_new['job tenure'], errors='coerce')
    df_new['job_stable'] = (df_new['job tenure'] >= 1).astype(int)  # 年資 >= 1 年

    print(f"  ✓ 工作穩定性 (年資 >= 1年)")
    print(f"    穩定客戶: {df_new['job_stable'].sum():,} ({df_new['job_stable'].mean()*100:.1f}%)")

    # ========== 聯絡穩定性指標 ==========
    print("\n【聯絡穩定性指標】")

    # 4. 戶籍居住一致性
    df_new['address_match'] = (
        df_new['post code of permanent address'] ==
        df_new['post code of residential address']
    ).astype(int)

    print(f"  ✓ 戶籍居住地一致")
    print(f"    一致比例: {df_new['address_match'].mean()*100:.1f}%")

    # 5. 居住狀況穩定性 (自有 > 租賃)
    stable_residence = ['自有', '配偶名下']
    df_new['residence_stable'] = df_new['residence status'].isin(stable_residence).astype(int)

    print(f"  ✓ 居住穩定性 (自有/配偶名下)")
    print(f"    穩定比例: {df_new['residence_stable'].mean()*100:.1f}%")

    # 6. 郵遞區號風險等級 (可用 WoE 轉換)
    # 暫時保留原始值，後續用 WoE 處理

    # ========== 逾期行為指標 ==========
    print("\n【逾期行為指標】")

    # 7. 總逾期次數
    overdue_cols = [
        'number of overdue before the first month',
        'number of overdue in the first half of the first month',
        'number of overdue in the second half of the first month',
    ]

    for col in overdue_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)

    df_new['total_overdue_count'] = df_new[overdue_cols].sum(axis=1)

    print(f"  ✓ 總逾期次數")
    print(f"    平均: {df_new['total_overdue_count'].mean():.2f} 次")
    print(f"    有逾期記錄: {(df_new['total_overdue_count']>0).sum():,} ({(df_new['total_overdue_count']>0).mean()*100:.1f}%)")

    # 8. 是否有逾期記錄
    df_new['has_overdue'] = (df_new['total_overdue_count'] > 0).astype(int)

    return df_new


def analyze_feature_importance_by_credit_logic(df):
    """
    根據徵審邏輯分析特徵重要性
    """

    print("\n" + "=" * 70)
    print("特徵重要性分析 (WoE + IV)")
    print("=" * 70)

    # 準備特徵
    # 財務能力
    financial_features = [
        'salary_level',           # 月薪等級
        'job_stable',             # 工作穩定性
    ]

    # 聯絡穩定性
    contact_features = [
        'post code of residential address',  # 居住郵遞區號
        'residence_stable',                   # 居住穩定性
        'address_match',                      # 戶籍居住一致
        'main business',                      # 主要經營業務
    ]

    # 逾期行為
    behavior_features = [
        'has_overdue',                        # 是否有逾期
    ]

    # 所有分類特徵
    categorical_features = financial_features + contact_features + behavior_features

    # 過濾存在的欄位
    categorical_features = [f for f in categorical_features if f in df.columns]

    # 分割數據
    X = df[categorical_features].copy()
    y = df['Default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # WoE 編碼
    woe_encoder = WoEEncoder()
    woe_encoder.fit(X_train, y_train, categorical_features)

    # 按重要性排序
    print("\n" + "=" * 70)
    print("特徵重要性排序 (徵審視角)")
    print("=" * 70)

    sorted_features = sorted(
        woe_encoder.iv_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n分類 | 特徵 | IV值 | 預測力")
    print("-" * 70)

    for feature, iv in sorted_features:
        # 判斷分類
        if feature in financial_features:
            category = "財務能力"
        elif feature in contact_features:
            category = "聯絡穩定"
        elif feature in behavior_features:
            category = "逾期行為"
        else:
            category = "其他"

        # 判斷預測力
        if iv > 0.3:
            strength = "強"
        elif iv > 0.1:
            strength = "中"
        elif iv > 0.02:
            strength = "弱"
        else:
            strength = "極弱"

        print(f"{category:10s} | {feature:40s} | {iv:.4f} | {strength}")

    return woe_encoder, categorical_features


def main():
    """主程序"""

    print("=" * 70)
    print("徵審導向的特徵分析")
    print("=" * 70)

    # 載入數據
    print("\n載入數據...")
    df = load_data()
    print(f"總樣本數: {len(df):,}")
    print(f"違約率: {df['Default'].mean():.2%}")

    # 創建衍生特徵
    df = create_derived_features(df)

    # 分析特徵重要性
    woe_encoder, categorical_features = analyze_feature_importance_by_credit_logic(df)

    # 數值特徵
    print("\n" + "=" * 70)
    print("數值特徵清單")
    print("=" * 70)

    numerical_features = [
        'payment_progress_ratio',   # 還款進度比率
        'month salary',             # 月薪
        'job tenure',               # 年資
        'total_overdue_count',      # 總逾期次數
        'loan term',                # 貸款期數
        'paid installments',        # 已繳期數
    ]

    numerical_features = [f for f in numerical_features if f in df.columns]

    print("\n可用數值特徵:")
    for feat in numerical_features:
        print(f"  - {feat}")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)
    print("\n重點發現:")
    print("1. 缺少「貸款總額」無法直接計算負債比")
    print("2. 使用「還款進度」作為財務能力代理指標")
    print("3. 郵遞區號可透過 WoE 轉換為風險評分")
    print("4. 逾期記錄是最強的預測因子")

    return df, woe_encoder, categorical_features, numerical_features


if __name__ == "__main__":
    df, woe_encoder, categorical_features, numerical_features = main()
