# -*- coding: utf-8 -*-
"""
DPM Prediction Module
=====================
使用訓練好的模型對新客戶進行違約風險預測

使用方式：
    python predict.py --input new_clients.xlsx --output predictions.xlsx
    或
    from predict import DPMPredictor
    predictor = DPMPredictor()
    predictions = predictor.predict(client_data)
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
import sys
sys.path.append('Train')
from feature_engineering import OverduePatternEncoder, get_feature_lists


class DPMPredictor:
    """
    DPM 違約預測器

    使用訓練好的模型對新客戶進行風險評估
    """

    def __init__(self, model_path='Train/models/best_model_stacking.pkl',
                 woe_encoder_path='Train/models/woe_encoder.pkl'):
        """
        初始化預測器

        Args:
            model_path: 訓練好的模型路徑
            woe_encoder_path: WoE 編碼器路徑
        """
        self.model_path = model_path
        self.woe_encoder_path = woe_encoder_path

        # 載入模型和編碼器
        self._load_model()
        self._load_woe_encoder()

        # 取得特徵列表
        self.categorical_features, self.numerical_features, _ = get_feature_lists()

        print("=" * 70)
        print("DPM Predictor Initialized")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"WoE Encoder: {woe_encoder_path}")
        print("=" * 70)

    def _load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        print(f"✓ Model loaded: {self.model_path}")

    def _load_woe_encoder(self):
        """載入 WoE 編碼器"""
        if not os.path.exists(self.woe_encoder_path):
            raise FileNotFoundError(f"WoE Encoder not found: {self.woe_encoder_path}")

        self.woe_encoder = joblib.load(self.woe_encoder_path)
        print(f"✓ WoE Encoder loaded: {self.woe_encoder_path}")

    def preprocess_data(self, df):
        """
        預處理新客戶資料

        Args:
            df: 新客戶資料 DataFrame

        Returns:
            X: 處理好的特徵矩陣
        """
        df = df.copy()

        print("\n[Preprocessing Data]")
        print(f"Input: {len(df)} clients, {len(df.columns)} columns")

        # 1. 處理缺失值
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
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 2. 轉換資料型態
        numeric_cols = ['loan term', 'paid installments', 'month salary',
                       'job tenure', 'overdue days']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. 特徵工程
        print("[Feature Engineering]")

        # 基本特徵
        df['payment_progress_ratio'] = (df['paid installments'] / df['loan term']).fillna(0).clip(0, 1)
        df['job_stable'] = (df['job tenure'] >= 1).astype(int)
        df['address_match'] = (
            df['post code of permanent address'] == df['post code of residential address']
        ).astype(int)

        # DTI features
        if 'debt_to_income_ratio' in df.columns:
            df['dti_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce').fillna(0).clip(0, 2)
        if 'payment_to_income_ratio' in df.columns:
            df['payment_pressure'] = pd.to_numeric(df['payment_to_income_ratio'], errors='coerce').fillna(0).clip(0, 1)

        # 早期逾期特徵
        early_overdue_cols = [
            'number of overdue before the first month',
            'number of overdue in the first half of the first month',
            'number of overdue in the second half of the first month',
        ]
        df['early_overdue_count'] = df[early_overdue_cols].sum(axis=1)
        df['has_overdue'] = (df['early_overdue_count'] > 0).astype(int)

        # Overdue Pattern Features
        overdue_encoder = OverduePatternEncoder()
        df = overdue_encoder.create_overdue_pattern_features(df)

        # 4. WoE 編碼
        print("[WoE Encoding]")
        categorical_features = [f for f in self.categorical_features if f in df.columns]
        numerical_features = [f for f in self.numerical_features if f in df.columns]

        # WoE transform
        df_woe = self.woe_encoder.transform(df[categorical_features], categorical_features)
        woe_cols = [f'{f}_WoE' for f in categorical_features if f'{f}_WoE' in df_woe.columns]

        # 組合特徵
        feature_cols = numerical_features + woe_cols
        feature_cols = [f for f in feature_cols if f in df.columns or f in df_woe.columns]

        # 建立最終特徵矩陣
        X = pd.DataFrame()
        for col in numerical_features:
            if col in df.columns:
                X[col] = df[col]
        for col in woe_cols:
            if col in df_woe.columns:
                X[col] = df_woe[col]

        X = X.fillna(0)

        print(f"Output: {X.shape[0]} clients, {X.shape[1]} features")

        return X

    def predict(self, df, return_proba=True):
        """
        預測新客戶的違約風險

        Args:
            df: 新客戶資料 DataFrame
            return_proba: 是否回傳機率 (True) 或類別 (False)

        Returns:
            如果 return_proba=True: 違約機率 (0-1)
            如果 return_proba=False: 違約預測 (0=正常, 1=違約)
        """
        # 預處理
        X = self.preprocess_data(df)

        # 預測
        print("\n[Predicting]")
        if return_proba:
            predictions = self.model.predict_proba(X)[:, 1]
            print(f"✓ Predicted default probabilities for {len(predictions)} clients")
        else:
            predictions = self.model.predict(X)
            print(f"✓ Predicted default classes for {len(predictions)} clients")

        return predictions

    def predict_with_details(self, df, threshold=0.5):
        """
        預測新客戶並提供詳細資訊

        Args:
            df: 新客戶資料 DataFrame
            threshold: 判定違約的閾值 (預設 0.5)

        Returns:
            result_df: 包含預測結果的 DataFrame
        """
        # 預測機率
        probas = self.predict(df, return_proba=True)

        # 建立結果 DataFrame
        result_df = df.copy()
        result_df['default_probability'] = probas
        result_df['predicted_default'] = (probas >= threshold).astype(int)
        result_df['risk_score'] = self._calculate_risk_score(probas)
        result_df['risk_grade'] = result_df['risk_score'].apply(self._get_risk_grade)
        result_df['decision'] = result_df['predicted_default'].apply(
            lambda x: 'REJECT' if x == 1 else 'APPROVE'
        )

        print("\n[Prediction Summary]")
        print("=" * 70)
        print(f"Total Clients: {len(result_df)}")
        print(f"Predicted Default: {result_df['predicted_default'].sum()} ({result_df['predicted_default'].mean():.1%})")
        print(f"Predicted Normal: {(result_df['predicted_default']==0).sum()} ({(result_df['predicted_default']==0).mean():.1%})")
        print("\nRisk Grade Distribution:")
        print(result_df['risk_grade'].value_counts().sort_index())
        print("=" * 70)

        return result_df

    def _calculate_risk_score(self, probabilities):
        """
        將違約機率轉換為風險分數 (0-100)
        機率越高，分數越低
        """
        # 反轉機率：1 - probability
        # 縮放到 0-100
        risk_score = (1 - probabilities) * 100
        return risk_score.astype(int)

    def _get_risk_grade(self, risk_score):
        """
        根據風險分數分級
        """
        if risk_score >= 80:
            return 'A (Excellent)'
        elif risk_score >= 60:
            return 'B (Good)'
        elif risk_score >= 40:
            return 'C (Fair)'
        elif risk_score >= 20:
            return 'D (Poor)'
        else:
            return 'E (High Risk)'


def main():
    """
    命令列介面
    """
    parser = argparse.ArgumentParser(description='DPM Default Prediction')
    parser.add_argument('--input', type=str, required=True,
                       help='Input Excel file with client data')
    parser.add_argument('--output', type=str, default='predictions.xlsx',
                       help='Output Excel file for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for default classification (default: 0.5)')
    parser.add_argument('--model', type=str, default='Train/models/best_model_stacking.pkl',
                       help='Path to trained model')
    parser.add_argument('--woe', type=str, default='Train/models/woe_encoder.pkl',
                       help='Path to WoE encoder')

    args = parser.parse_args()

    print("=" * 80)
    print("DPM Default Prediction System")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)

    # 載入資料
    print(f"\n[Loading Data]")
    df = pd.read_excel(args.input, engine='openpyxl')
    print(f"✓ Loaded {len(df)} clients from {args.input}")

    # 初始化預測器
    predictor = DPMPredictor(model_path=args.model, woe_encoder_path=args.woe)

    # 預測
    result_df = predictor.predict_with_details(df, threshold=args.threshold)

    # 儲存結果
    print(f"\n[Saving Results]")
    result_df.to_excel(args.output, index=False, engine='openpyxl')
    print(f"✓ Predictions saved to {args.output}")

    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
