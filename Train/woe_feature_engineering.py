# -*- coding: utf-8 -*-
"""
WoE Feature Engineering for DPM
借用東吳 WoE 技術來增強 DPM 模型的特徵工程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class WoEEncoder:
    """
    Weight of Evidence (WoE) 編碼器
    用於將分類變數轉換為 WoE 值
    """

    def __init__(self):
        self.woe_dict = {}
        self.iv_dict = {}

    def calculate_woe_iv(self, df, feature, target='Default'):
        """
        計算 WoE 和 IV 值

        Args:
            df: DataFrame
            feature: 特徵名稱
            target: 目標變數名稱

        Returns:
            woe_df: 包含 WoE 和 IV 的 DataFrame
        """
        # 計算每個類別的好壞客戶數量
        grouped = df.groupby(feature)[target].agg(['sum', 'count'])
        grouped.columns = ['bad', 'total']
        grouped['good'] = grouped['total'] - grouped['bad']

        # 計算總好壞客戶數
        total_good = grouped['good'].sum()
        total_bad = grouped['bad'].sum()

        # 避免除以零
        if total_good == 0 or total_bad == 0:
            print(f"[WARNING] {feature}: 沒有足夠的好壞樣本，跳過")
            return None

        # 計算每個類別的好壞比例
        grouped['good_pct'] = grouped['good'] / total_good
        grouped['bad_pct'] = grouped['bad'] / total_bad

        # 避免 0 值導致 log 錯誤
        grouped['good_pct'] = grouped['good_pct'].replace(0, 0.0001)
        grouped['bad_pct'] = grouped['bad_pct'].replace(0, 0.0001)

        # 計算 WoE
        grouped['WoE'] = np.log(grouped['good_pct'] / grouped['bad_pct'])

        # 計算 IV
        grouped['IV'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['WoE']

        # 重設索引
        grouped = grouped.reset_index()

        return grouped

    def fit(self, X, y, categorical_features):
        """
        訓練 WoE 編碼器

        Args:
            X: 特徵 DataFrame
            y: 目標變數 Series
            categorical_features: 分類特徵列表
        """
        # 合併 X 和 y
        df = X.copy()
        df['Default'] = y

        print("=" * 70)
        print("計算 WoE 和 IV 值")
        print("=" * 70)

        for feature in categorical_features:
            if feature not in df.columns:
                print(f"[WARNING] {feature} 不存在，跳過")
                continue

            # 計算 WoE 和 IV
            woe_df = self.calculate_woe_iv(df, feature, 'Default')

            if woe_df is not None:
                # 儲存 WoE 對照表
                self.woe_dict[feature] = dict(zip(woe_df[feature], woe_df['WoE']))

                # 儲存 IV 值
                total_iv = woe_df['IV'].sum()
                self.iv_dict[feature] = total_iv

                # 顯示 IV 值
                if total_iv > 0.3:
                    strength = "強預測力"
                elif total_iv > 0.1:
                    strength = "中等預測力"
                elif total_iv > 0.02:
                    strength = "弱預測力"
                else:
                    strength = "幾乎無預測力"

                print(f"\n{feature}:")
                print(f"  IV = {total_iv:.4f} ({strength})")
                print(f"  類別數量: {len(woe_df)}")

        # 按 IV 值排序
        print("\n" + "=" * 70)
        print("特徵重要性排序 (依 IV 值)")
        print("=" * 70)
        sorted_iv = sorted(self.iv_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, iv in sorted_iv:
            print(f"{feature:30s} | IV = {iv:.4f}")

    def transform(self, X, categorical_features):
        """
        將特徵轉換為 WoE 值

        Args:
            X: 特徵 DataFrame
            categorical_features: 分類特徵列表

        Returns:
            X_woe: WoE 轉換後的 DataFrame
        """
        X_woe = X.copy()

        for feature in categorical_features:
            if feature not in self.woe_dict:
                continue

            # 轉換為 WoE 值
            woe_map = self.woe_dict[feature]
            X_woe[f'{feature}_WoE'] = X[feature].map(woe_map)

            # 處理未見過的類別（設為 0）
            X_woe[f'{feature}_WoE'] = X_woe[f'{feature}_WoE'].fillna(0)

        return X_woe

    def fit_transform(self, X, y, categorical_features):
        """訓練並轉換"""
        self.fit(X, y, categorical_features)
        return self.transform(X, categorical_features)

    def get_important_features(self, iv_threshold=0.02):
        """
        根據 IV 值篩選重要特徵

        Args:
            iv_threshold: IV 閾值

        Returns:
            important_features: 重要特徵列表
        """
        important_features = [
            feature for feature, iv in self.iv_dict.items()
            if iv >= iv_threshold
        ]
        return important_features


def demo_woe_feature_engineering():
    """示範如何使用 WoE 特徵工程"""

    print("=" * 70)
    print("WoE 特徵工程示範")
    print("=" * 70)

    # 1. 載入數據
    print("\n1. 載入數據...")
    df = pd.read_excel('source/DPM_data_cleaned.xlsx', engine='openpyxl')

    # 2. 定義違約（使用 overdue status）
    print("\n2. 定義違約標籤...")
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
    print(f"   違約率: {df['Default'].mean():.2%}")

    # 3. 定義分類特徵
    categorical_features = [
        'education',           # 教育程度
        'residence status',    # 居住狀況
        'product',            # 產品別
        'marriage',           # 婚姻
        'main business',      # 主要經營業務
    ]

    # 4. 準備數據
    feature_cols = categorical_features
    X = df[feature_cols].copy()
    y = df['Default']

    # 5. 訓練-測試分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. WoE 編碼
    print("\n3. 訓練 WoE 編碼器...")
    woe_encoder = WoEEncoder()
    X_train_woe = woe_encoder.fit_transform(X_train, y_train, categorical_features)
    X_test_woe = woe_encoder.transform(X_test, categorical_features)

    # 7. 篩選重要特徵
    print("\n4. 篩選重要特徵 (IV > 0.02)...")
    important_features = woe_encoder.get_important_features(iv_threshold=0.02)
    print(f"\n重要特徵數量: {len(important_features)}")
    for feat in important_features:
        print(f"  - {feat} (IV = {woe_encoder.iv_dict[feat]:.4f})")

    # 8. 顯示轉換後的數據
    print("\n5. WoE 轉換結果預覽:")
    woe_columns = [col for col in X_train_woe.columns if '_WoE' in col]
    print(X_train_woe[woe_columns].head())

    print("\n" + "=" * 70)
    print("完成！WoE 特徵已準備好，可以加入 DPM 模型訓練")
    print("=" * 70)

    return woe_encoder, X_train_woe, X_test_woe, y_train, y_test


if __name__ == "__main__":
    # 執行示範
    woe_encoder, X_train_woe, X_test_woe, y_train, y_test = demo_woe_feature_engineering()
