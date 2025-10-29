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
import json
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Train'))
from feature_engineering import WoEEncoder, CustomerHistoryEncoder, OverduePatternEncoder, get_feature_lists


class DPMPredictor:
    """
    DPM 違約預測器

    使用訓練好的模型對新客戶進行風險評估
    """

    def __init__(self, model_path='../Train/models/best_model_stacking.pkl',
                 woe_encoder_path='../Train/models/woe_encoder.pkl',
                 threshold_config_path='../Train/models/optimal_thresholds.json'):
        """
        初始化預測器

        Args:
            model_path: 訓練好的模型路徑
            woe_encoder_path: WoE 編碼器路徑
            threshold_config_path: 最佳閾值配置檔路徑
        """
        self.model_path = model_path
        self.woe_encoder_path = woe_encoder_path
        self.threshold_config_path = threshold_config_path

        # 載入模型和編碼器
        self._load_model()
        self._load_woe_encoder()

        # 自動載入最佳閾值
        self.optimal_threshold = self._load_optimal_threshold()

        # 取得特徵列表
        self.categorical_features, self.numerical_features, _ = get_feature_lists()

        print("=" * 70)
        print("DPM Predictor Initialized")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"WoE Encoder: {woe_encoder_path}")
        print(f"Optimal Threshold: {self.optimal_threshold:.3f}")
        print("=" * 70)

    def _load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        print(f"[OK] Model loaded: {self.model_path}")

    def _load_woe_encoder(self):
        """載入 WoE 編碼器"""
        if not os.path.exists(self.woe_encoder_path):
            raise FileNotFoundError(f"WoE Encoder not found: {self.woe_encoder_path}")

        self.woe_encoder = joblib.load(self.woe_encoder_path)
        print(f"[OK] WoE Encoder loaded: {self.woe_encoder_path}")

    def _load_optimal_threshold(self):
        """
        自動載入最佳閾值

        Returns:
            float: 最佳閾值，如果 JSON 不存在或載入失敗則使用預設值 0.5（標準閾值）
        """
        default_threshold = 0.5  # 預設值：標準統計閾值（找不到訓練配置時使用）

        try:
            if os.path.exists(self.threshold_config_path):
                with open(self.threshold_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    threshold = config.get('recommended_threshold', default_threshold)
                    trained_date = config.get('trained_date', 'Unknown')

                    print(f"[OK] Loaded optimal threshold: {threshold:.3f}")
                    print(f"     Trained date: {trained_date}")

                    # 顯示預期效能
                    if 'Stacking' in config.get('optimal_thresholds', {}):
                        stacking_info = config['optimal_thresholds']['Stacking']
                        recall = stacking_info.get('recall', 0)
                        precision = stacking_info.get('precision', 0)
                        print(f"     Expected Recall: {recall*100:.2f}%")
                        print(f"     Expected Precision: {precision*100:.2f}%")

                    return threshold
            else:
                print(f"[WARNING] Threshold config not found: {self.threshold_config_path}")
                print(f"          Using default threshold: {default_threshold:.3f} (standard statistical threshold)")
                print(f"          Please run training to generate optimal threshold configuration")
                return default_threshold

        except Exception as e:
            print(f"[ERROR] Failed to load threshold config: {e}")
            print(f"        Using default threshold: {default_threshold:.3f} (standard statistical threshold)")
            return default_threshold

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

        # Residence stability
        if 'residence status' in df.columns:
            df['residence_stable'] = df['residence status'].isin(['own', 'spouse', 'family']).astype(int)
        else:
            df['residence_stable'] = 0

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

        # Customer History Features
        # NOTE: 訓練時沒有使用 Customer History 特徵，所以預測時也不產生
        # 如果未來重新訓練時有包含這些特徵，再啟用此段
        # if 'ID' in df.columns and 'application date' in df.columns:
        #     print("[Customer History Encoding]")
        #     customer_encoder = CustomerHistoryEncoder()
        #     df = customer_encoder.fit_transform(df)
        #     print(f"[OK] Customer history features created")
        # else:
        #     df['is_new_customer'] = 1
        #     df['historical_default'] = 0
        #     df['prev_overdue_days'] = 0
        #     df['prev_payment_ratio'] = 0
        #     df['cumulative_loans'] = 1
        #     df['cumulative_defaults'] = 0
        #     df['customer_default_rate'] = 0
        #     print("[OK] Treated all as new customers (no historical data)")
        print("[OK] Skipped customer history features (not used in training)")

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
            print(f"[OK] Predicted default probabilities for {len(predictions)} clients")
        else:
            predictions = self.model.predict(X)
            print(f"[OK] Predicted default classes for {len(predictions)} clients")

        return predictions

    def predict_with_details(self, df, threshold=0.5, simplified_output=False):
        """
        預測新客戶並提供詳細資訊

        Args:
            df: 新客戶資料 DataFrame
            threshold: 判定違約的閾值 (預設 0.5，用於標準比較)
            simplified_output: 是否只輸出簡化欄位 (預設 False)

        Returns:
            result_df: 包含預測結果的 DataFrame
        """
        # 預測機率
        probas = self.predict(df, return_proba=True)

        # 使用自動載入的最佳閾值
        optimal_threshold = self.optimal_threshold

        # 建立結果 DataFrame
        result_df = df.copy()
        result_df['default_probability'] = (probas * 100).round(2)  # 轉換為百分比並取到小數點後2位
        result_df['predicted_default'] = (probas >= threshold).astype(int)
        result_df['predicted_default_optimal'] = (probas >= optimal_threshold).astype(int)
        result_df['threshold_difference'] = (result_df['predicted_default'] != result_df['predicted_default_optimal']).astype(int)
        result_df['risk_score'] = self._calculate_risk_score(probas)
        result_df['risk_grade'] = result_df['risk_score'].apply(self._get_risk_grade)
        result_df['risk_alert'] = result_df['risk_score'].apply(self._get_risk_alert)
        result_df['risk_action'] = result_df.apply(self._get_risk_action, axis=1)
        result_df['risk_action_optimal'] = result_df.apply(self._get_risk_action_optimal, axis=1)

        print("\n[Prediction Summary]")
        print("=" * 70)
        print(f"Total Clients: {len(result_df)}")
        print(f"\n[Standard Threshold {threshold:.3f}]")
        print(f"Predicted Default: {result_df['predicted_default'].sum()} ({result_df['predicted_default'].mean():.1%})")
        print(f"Predicted Normal: {(result_df['predicted_default']==0).sum()} ({(result_df['predicted_default']==0).mean():.1%})")
        print(f"\n[Optimal Threshold {optimal_threshold:.3f}]")
        print(f"Predicted Default: {result_df['predicted_default_optimal'].sum()} ({result_df['predicted_default_optimal'].mean():.1%})")
        print(f"Predicted Normal: {(result_df['predicted_default_optimal']==0).sum()} ({(result_df['predicted_default_optimal']==0).mean():.1%})")
        print(f"\n[Threshold Difference]")
        print(f"Cases with Different Prediction: {result_df['threshold_difference'].sum()} ({result_df['threshold_difference'].mean():.1%})")
        print("\nRisk Grade Distribution:")
        print(result_df['risk_grade'].value_counts().sort_index())
        print("\n[Risk Action - Standard Threshold]")
        print(result_df['risk_action'].value_counts())
        print("\n[Risk Action - Optimal Threshold]")
        print(result_df['risk_action_optimal'].value_counts())
        print("=" * 70)

        # 如果需要簡化輸出，只保留關鍵欄位
        if simplified_output:
            result_df = self._create_simplified_output(result_df, threshold)

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
        根據風險分數分級（符合業界標準）

        分級標準動態對齊最佳閾值：
        - A (優良): 違約機率 < optimal_threshold (LOW RISK)
        - B (中等): 違約機率 optimal_threshold ~ 40% (MEDIUM-HIGH RISK)
        - C (警戒): 違約機率 41-60% (HIGH RISK)
        - D (不良): 違約機率 61-80% (VERY HIGH RISK)
        - E (危險): 違約機率 81-100% (CRITICAL RISK)
        """
        # 計算 A 級的分數閾值（動態基於 optimal_threshold）
        # risk_score = (1 - probability) * 100
        # 所以 probability = optimal_threshold 時，risk_score = (1 - optimal_threshold) * 100
        threshold_score = (1 - self.optimal_threshold) * 100

        if risk_score >= threshold_score:  # 違約機率 < optimal_threshold
            return 'A (優良)'
        elif risk_score >= 60:  # 違約機率 optimal_threshold ~ 40%
            return 'B (中等)'
        elif risk_score >= 40:  # 違約機率 41-60%
            return 'C (警戒)'
        elif risk_score >= 20:  # 違約機率 61-80%
            return 'D (不良)'
        else:  # 違約機率 81-100%
            return 'E (危險)'

    def _get_risk_alert(self, risk_score):
        """
        根據風險分數給出多級警示

        5級警示系統（動態對齊最佳閾值）：
        - LOW RISK: A級 (違約機率 < optimal_threshold)
        - MEDIUM-HIGH RISK: B級 (違約機率 optimal_threshold ~ 40%)
        - HIGH RISK: C級 (違約機率 41-60%)
        - VERY HIGH RISK: D級 (違約機率 61-80%)
        - CRITICAL RISK: E級 (違約機率 81-100%)
        """
        # 使用與 _get_risk_grade 相同的動態閾值
        threshold_score = (1 - self.optimal_threshold) * 100

        if risk_score >= threshold_score:
            return 'LOW RISK'
        elif risk_score >= 60:
            return 'MEDIUM-HIGH RISK'
        elif risk_score >= 40:
            return 'HIGH RISK'
        elif risk_score >= 20:
            return 'VERY HIGH RISK'
        else:
            return 'CRITICAL RISK'

    def _get_risk_action(self, row):
        """
        根據風險等級和預警指標決定管理動作

        風險管理動作：
        - 正常監控: A級
        - 關注名單: B級（正常）
        - 早期預警: B級 + 早期預警標記
        - 加強監控: C級（正常）
        - 惡化警示: C級 + 惡化趨勢
        - 立即介入: D級
        - 強制催收: E級
        """
        grade = row['risk_grade']
        early_warning = row.get('early_warning_flag', 0)
        worsening = row.get('overdue_worsening', 0)

        if grade == 'A (優良)':
            return '正常監控'
        elif grade == 'B (中等)':
            if early_warning == 1:
                return '早期預警'
            else:
                return '關注名單'
        elif grade == 'C (警戒)':
            if worsening == 1:
                return '惡化警示'
            else:
                return '加強監控'
        elif grade == 'D (不良)':
            return '立即介入'
        else:  # E (危險)
            return '強制催收'

    def _get_risk_action_optimal(self, row):
        """
        根據風險等級、預警指標和最佳閾值決定管理動作（動態基於 optimal_threshold）

        當 predicted_default_optimal=1 時，管理動作會升級：
        - A級 + 高風險 → 關注名單（從正常監控升級）
        - B級 + 高風險 → 加強監控/催收準備（從關注名單升級）
        - C級 + 高風險 → 立即介入/催收準備（從加強監控升級）
        - D級 + 高風險 → 強制催收（從立即介入升級）
        - E級 → 強制催收（已是最高級）
        """
        grade = row['risk_grade']
        early_warning = row.get('early_warning_flag', 0)
        worsening = row.get('overdue_worsening', 0)
        optimal_high_risk = row.get('predicted_default_optimal', 0)

        if grade == 'A (優良)':
            if optimal_high_risk == 1:
                return '關注名單'  # 升級
            else:
                return '正常監控'
        elif grade == 'B (中等)':
            if optimal_high_risk == 1:
                if early_warning == 1:
                    return '加強監控+早期預警'  # 升級
                else:
                    return '加強監控/催收準備'  # 升級
            else:
                if early_warning == 1:
                    return '早期預警'
                else:
                    return '關注名單'
        elif grade == 'C (警戒)':
            if optimal_high_risk == 1:
                if worsening == 1:
                    return '立即催收'  # 升級
                else:
                    return '立即介入/催收準備'  # 升級
            else:
                if worsening == 1:
                    return '惡化警示'
                else:
                    return '加強監控'
        elif grade == 'D (不良)':
            if optimal_high_risk == 1:
                return '強制催收'  # 升級
            else:
                return '立即介入'
        else:  # E (危險)
            return '強制催收'

    def _create_simplified_output(self, df, threshold):
        """
        建立簡化版輸出（只保留關鍵欄位）

        Args:
            df: 完整預測結果 DataFrame
            threshold: 使用的閾值

        Returns:
            simplified_df: 只包含關鍵欄位的 DataFrame
        """
        # 定義要保留的欄位（按照使用者指定的順序）
        keep_columns = []

        # 基本識別欄位（處理可能的換行符號）
        serial_col = None
        for col in df.columns:
            if 'serial number' in col.lower().strip():
                serial_col = col
                break
        if serial_col:
            keep_columns.append(serial_col)

        if 'application date' in df.columns:
            keep_columns.append('application date')
        if 'applicant' in df.columns:
            keep_columns.append('applicant')
        if 'ID' in df.columns:
            keep_columns.append('ID')

        # 歷史資料欄位（重新命名）
        if 'default rate' in df.columns:
            df['historical_default_rate'] = df['default rate']
            keep_columns.append('historical_default_rate')

        # 預測結果欄位（按照邏輯順序排列）
        keep_columns.extend([
            'default_probability',
            'risk_score',
            'risk_grade',
            'risk_action',
            'risk_action_optimal',
            'risk_alert',
            'predicted_default',
            'predicted_default_optimal',
            'threshold_difference'
        ])

        # 如果有預警特徵，也加入
        if 'early_warning_flag' in df.columns:
            keep_columns.append('early_warning_flag')
        if 'overdue_worsening' in df.columns:
            keep_columns.append('overdue_worsening')

        # 只保留存在的欄位
        final_columns = [col for col in keep_columns if col in df.columns]
        simplified_df = df[final_columns].copy()

        # 加入中文欄位說明（作為第一行）
        column_descriptions = {
            'application date': '申請日期',
            'applicant': '申請人姓名',
            'ID': '身分證字號',
            'historical_default_rate': '歷史違約率(%) - 此客戶過去的違約記錄，僅供參考',
            'default_probability': f'違約機率(%) - 模型預測此客戶未來違約的機率',
            'risk_score': '風險分數 - 0-100分，分數越高越安全(100分=0%違約機率)',
            'risk_grade': '風險等級 - A(優良)最好, E(危險)最差，與業界標準對齊',
            'risk_action': f'風險管理動作(標準閾值{threshold:.3f}) - 根據風險等級與預警指標建議的管理動作',
            'risk_action_optimal': f'風險管理動作(最佳閾值{self.optimal_threshold:.3f}) - 當高風險標記=1時，管理動作會升級（如B級從關注名單→加強監控/催收準備）',
            'risk_alert': '風險警示 - 5級警示系統：LOW RISK(低風險) / MEDIUM-HIGH RISK(中高風險) / HIGH RISK(高風險) / VERY HIGH RISK(極高風險) / CRITICAL RISK(危急風險)',
            'predicted_default': f'高風險標記(標準閾值{threshold:.3f}) - 0=低風險, 1=高風險',
            'predicted_default_optimal': f'高風險標記(最佳閾值{self.optimal_threshold:.3f}) - 0=低風險, 1=高風險',
            'threshold_difference': '閾值差異 - 1=兩種閾值判斷不同, 0=判斷相同',
            'early_warning_flag': '早期預警標記 - 1=有早期預警跡象, 0=正常',
            'overdue_worsening': '惡化趨勢標記 - 1=逾期情況惡化中, 0=正常或改善'
        }

        # 為 serial number 加入說明（不管原始名稱）
        if serial_col and serial_col in simplified_df.columns:
            column_descriptions[serial_col] = '案件編號'

        # 將欄位說明加入 DataFrame 的 metadata（透過註解）
        simplified_df.attrs['column_descriptions'] = column_descriptions

        return simplified_df


def list_available_files():
    """列出可用的資料檔案"""
    files = []
    search_paths = [
        ('Train/Source', '../Train/Source'),
        ('Prediction/Source', 'Source'),
        ('Prediction Data', 'Prediction data'),
        ('Result (歷史)', 'Result')
    ]

    print("\n" + "=" * 80)
    print("可用的資料檔案：")
    print("=" * 80)

    idx = 1
    for label, path in search_paths:
        if os.path.exists(path):
            excel_files = [f for f in os.listdir(path) if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
            if excel_files:
                print(f"\n【{label}】")
                for file in sorted(excel_files):
                    full_path = os.path.join(path, file)
                    try:
                        file_size = os.path.getsize(full_path) / 1024
                        print(f"  {idx}. {file} ({file_size:.1f} KB)")
                        files.append((idx, full_path, file))
                        idx += 1
                    except:
                        pass
    return files


def interactive_mode():
    """互動模式：引導使用者選擇檔案"""
    print("=" * 80)
    print("DPM 互動式預測工具")
    print("=" * 80)

    # 列出可用檔案
    available_files = list_available_files()

    if not available_files:
        print("\n[ERROR] 找不到任何 Excel 檔案")
        return None, None

    # 讓使用者選擇
    print("\n" + "=" * 80)
    print("請選擇要預測的檔案：")
    print("  - 輸入編號 (例如：1)")
    print("  - 或輸入完整路徑")
    print("  - 輸入 'q' 離開")
    print("=" * 80)

    choice = input("\n請輸入: ").strip()

    if choice.lower() == 'q':
        return None, None

    # 處理選擇
    input_file = None
    try:
        file_idx = int(choice)
        for idx, path, name in available_files:
            if idx == file_idx:
                input_file = path
                break
    except ValueError:
        if os.path.exists(choice):
            input_file = choice

    if not input_file:
        print(f"[ERROR] 無效的選擇")
        return None, None

    print(f"✓ 已選擇: {input_file}")

    # 詢問輸出檔名
    default_output = f'Result/Predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    print(f"\n輸出檔案 (預設: {default_output})")
    custom_output = input("自訂檔名 (Enter=使用預設): ").strip()
    output_file = custom_output if custom_output else default_output

    # 確認
    print(f"\n輸入: {input_file}")
    print(f"輸出: {output_file}")
    confirm = input("確認執行? (y/n): ").strip().lower()

    if confirm != 'y':
        return None, None

    return input_file, output_file


def export_to_csv(result_df, output_path, predictor):
    """
    匯出 CSV 格式（簡化版，供資料庫匯入）

    只輸出 9 個關鍵欄位，使用中英文雙語標題

    Args:
        result_df: 完整預測結果 DataFrame
        output_path: CSV 輸出路徑
        predictor: DPMPredictor 實例（用來取得 optimal_threshold）
    """

    # 選取 9 個欄位（對應 A/B/C/D/F/G/H/J/K）
    csv_columns_mapping = {
        'serial number': 'serial number (流水號)',
        'application date': 'application date (申請日期)',
        'applicant': 'applicant (申請人)',
        'ID': 'ID (身分證字號)',
        'default_probability': 'default_probability (違約機率)',
        'risk_score': 'risk_score (風險分數)',
        'risk_grade': 'risk_grade (風險等級)',
        'risk_action_optimal': 'risk_action_optimal (風險管理動作)',
        'risk_alert': 'risk_alert (風險警示)'
    }

    # 只保留存在的欄位
    available_columns = [col for col in csv_columns_mapping.keys() if col in result_df.columns]

    if len(available_columns) == 0:
        print("[WARNING] No columns available for CSV export")
        return

    # 建立 CSV DataFrame
    result_df_csv = result_df[available_columns].copy()

    # 重新命名為中英文雙語標題
    result_df_csv.columns = [csv_columns_mapping[col] for col in available_columns]

    # 輸出 CSV（使用 utf-8-sig 讓 Excel 正確顯示中文）
    result_df_csv.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"     CSV 輸出 {len(result_df_csv)} 筆記錄，{len(result_df_csv.columns)} 個欄位")


def generate_readme(output_path, predictor):
    """
    產生評分規則說明文件

    Args:
        output_path: README 輸出路徑
        predictor: DPMPredictor 實例
    """

    # 取得閾值相關資訊
    optimal_threshold = predictor.optimal_threshold
    threshold_score = int((1 - optimal_threshold) * 100)
    optimal_threshold_pct = optimal_threshold * 100

    # 讀取訓練日期（從 JSON 檔）
    trained_date = "Unknown"
    try:
        if os.path.exists(predictor.threshold_config_path):
            with open(predictor.threshold_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                trained_date = config.get('trained_date', 'Unknown')
    except:
        pass

    # 產生說明文件內容
    readme_content = f"""DPM 違約預測模型 - B Card 風險評估說明
{'=' * 80}

模型資訊
--------
訓練日期：{trained_date}
最佳閾值：{optimal_threshold:.3f} (F2-score 優化，提高違約偵測率)
模型類型：Stacking Ensemble (LightGBM + XGBoost + CatBoost)
應用場景：B Card (貸後管理) - 已放款客戶的違約風險監控

檔案說明
--------
本 CSV 檔案包含 9 個欄位，供資料庫匯入使用：

1. serial number (流水號)         - 案件編號
2. application date (申請日期)     - 貸款申請日期
3. applicant (申請人)             - 申請人姓名
4. ID (身分證字號)                - 申請人身分證字號
5. default_probability (違約機率)  - 模型預測的違約機率 (0-1)
6. risk_score (風險分數)          - 風險評分 (0-100)
7. risk_grade (風險等級)          - 風險等級分類 (A-E)
8. risk_action_optimal (風險管理動作) - 催收管理策略建議
9. risk_alert (風險警示)          - 風險警示等級

評分計算方式
------------
1. 違約機率 (default_probability)
   - 由 Stacking 模型預測輸出
   - 範圍：0-1 (0 = 不會違約, 1 = 必定違約)
   - 用途：評估已放款客戶的違約風險

2. 風險分數 (risk_score)
   - 計算公式：risk_score = (1 - default_probability) × 100
   - 範圍：0-100 分 (分數越高越安全)
   - 例如：違約機率 0.08 → 風險分數 92 分

風險等級分類標準 (動態基於最佳閾值 {optimal_threshold:.3f})
{'=' * 80}

等級 A (優良) - 信用優良客戶
  Risk Score：{threshold_score}-100 分
  違約機率：0-{optimal_threshold_pct:.0f}%
  風險警示：LOW RISK
  管理策略：正常管理
  說明：違約機率低於最佳閾值，極低風險客戶，維持正常聯繫頻率即可

等級 B (中等) - 信用良好客戶
  Risk Score：60-{threshold_score-1} 分
  違約機率：{optimal_threshold_pct+1:.0f}-40%
  風險警示：MEDIUM-HIGH RISK
  管理策略：加強監控/催收準備
  說明：中等風險客戶，建議提高聯繫頻率，注意早期惡化跡象

等級 C (警戒) - 需要關注客戶
  Risk Score：40-59 分
  違約機率：41-60%
  風險警示：HIGH RISK
  管理策略：強制催收
  說明：高風險客戶，立即啟動催收程序，密切追蹤還款狀況

等級 D (不良) - 高風險客戶
  Risk Score：20-39 分
  違約機率：61-80%
  風險警示：HIGH RISK
  管理策略：強制催收
  說明：嚴重違約風險，需採取積極催收措施，考慮法律途徑

等級 E (危險) - 違約臨界客戶
  Risk Score：0-19 分
  違約機率：81-100%
  風險警示：CRITICAL RISK
  管理策略：強制催收
  說明：違約臨界狀態，幾乎確定違約，緊急啟動催收流程並評估損失

風險管理動作 (risk_action_optimal)
{'=' * 80}
根據最佳閾值 {optimal_threshold:.3f} 的催收管理策略：

- 正常管理：違約機率 < {optimal_threshold_pct:.0f}% (低於最佳閾值)
  → 低風險客戶，維持正常聯繫與追蹤

- 加強監控/催收準備：違約機率 {optimal_threshold_pct:.0f}-40%
  → 中等風險客戶，需加強聯繫頻率，提早預警

- 強制催收：違約機率 > 40%
  → 高風險客戶，立即啟動催收程序，防止損失擴大

風險警示 (risk_alert)
{'=' * 80}
- LOW RISK：違約機率 < {optimal_threshold_pct:.0f}%
  → 正常狀態，持續追蹤即可

- MEDIUM-HIGH RISK：違約機率 {optimal_threshold_pct:.0f}-40%
  → 早期惡化警示，需提高關注度與聯繫頻率

- HIGH RISK：違約機率 40-80%
  → 嚴重惡化警示，立即介入處理

- CRITICAL RISK：違約機率 > 80%
  → 違約臨界狀態，緊急啟動催收流程

使用說明
------------
1. 模型用途：使用已訓練完成的 DPM 模型，對現有貸款案件進行違約風險評估
2. 應用場景：B Card（貸後管理），用於監控已放款客戶的違約風險
3. 管理策略：根據預測結果採取差異化催收管理，提早偵測風險惡化
4. 閾值設定：採用 F2-score 優化的最佳閾值（提高違約偵測率，降低漏報）
5. 動態更新：風險等級分類標準隨最佳閾值動態調整
6. 綜合判斷：建議結合客戶歷史行為、還款記錄等資訊綜合評估
7. 技術支援：如有疑問請聯繫模型維護人員

{'=' * 80}
Generated by DPM Prediction System (B Card)
{'=' * 80}
"""

    # 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)


def main():
    """
    命令列介面
    """
    parser = argparse.ArgumentParser(description='DPM Default Prediction')
    parser.add_argument('--input', type=str, required=False,
                       help='Input Excel file with client data (if not provided, enters interactive mode)')
    parser.add_argument('--output', type=str, default='predictions.xlsx',
                       help='Output Excel file for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for default classification (default: 0.5)')
    parser.add_argument('--model', type=str, default='../Train/models/best_model_stacking.pkl',
                       help='Path to trained model')
    parser.add_argument('--woe', type=str, default='../Train/models/woe_encoder.pkl',
                       help='Path to WoE encoder')
    parser.add_argument('--simplified', action='store_true',
                       help='Output simplified format with key columns only')

    args = parser.parse_args()

    # 如果沒有提供 input，進入互動模式
    if not args.input:
        input_file, output_file = interactive_mode()
        if not input_file:
            print("已取消")
            return
        args.input = input_file
        args.output = output_file

    print("\n" + "=" * 80)
    print("DPM Default Prediction System")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Simplified Output: {args.simplified}")
    print("=" * 80)

    # 載入資料
    print(f"\n[Loading Data]")
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input, engine='openpyxl')
    print(f"[OK] Loaded {len(df)} clients from {args.input}")

    # 初始化預測器
    predictor = DPMPredictor(model_path=args.model, woe_encoder_path=args.woe)

    # 預測
    result_df = predictor.predict_with_details(df, threshold=args.threshold,
                                               simplified_output=args.simplified)

    # 儲存結果
    print(f"\n[Saving Results]")

    # 統一欄位名稱：將 'default rate' 改名為 'historical_default_rate'
    if 'default rate' in result_df.columns and 'historical_default_rate' not in result_df.columns:
        result_df = result_df.rename(columns={'default rate': 'historical_default_rate'})

    # 只保留關鍵欄位（按照定案的格式）
    key_columns = [
        '\nserial number',         # 案件編號（注意：原始資料可能有換行符號）
        'serial number',           # 也支援沒有換行符號的版本
        'application date',        # 申請日期
        'applicant',              # 申請人
        'ID',                     # 身分證字號
        'historical_default_rate', # 歷史違約率（政大評分）
        'default_probability',     # 違約機率
        'risk_score',             # 風險分數
        'risk_grade',             # 風險等級
        'risk_action',            # 風險管理動作（標準閾值）
        'risk_action_optimal',    # 風險管理動作（最佳閾值）
        'risk_alert',             # 風險警示
        'predicted_default',      # 高風險標記（標準）
        'predicted_default_optimal', # 高風險標記（最佳）
        'threshold_difference'    # 閾值差異
    ]

    # 只保留存在的欄位（去重）
    output_columns = []
    seen = set()
    for col in key_columns:
        if col in result_df.columns and col not in seen:
            output_columns.append(col)
            seen.add(col)

    result_df_output = result_df[output_columns].copy()

    # 清理欄位名稱（移除換行符號）
    result_df_output.columns = [col.strip() for col in result_df_output.columns]

    print(f"輸出欄位數: {len(output_columns)} (原始: {len(result_df.columns)})")

    # 預設使用簡化版（多頁籤）
    if not args.simplified:
        args.simplified = True  # 強制使用簡化版

    if args.simplified:
        # 簡化版：輸出多個 Sheet（預測結果 + 欄位說明）
        with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
            # Sheet 1: 預測結果（只輸出關鍵欄位）
            result_df_output.to_excel(writer, sheet_name='預測結果', index=False)

            # 計算閾值相關數值（所有 Sheet 共用）
            optimal_threshold_pct = predictor.optimal_threshold * 100  # 轉為百分比
            threshold_score = int((1 - predictor.optimal_threshold) * 100)  # 計算分數閾值

            # Sheet 2: 欄位說明
            # 定義所有欄位的說明
            field_descriptions = {
                'serial number': '案件編號',
                'application date': '申請日期',
                'applicant': '申請人姓名',
                'ID': '身分證字號',
                'historical_default_rate': '歷史違約率（來自政大評分）',
                'default_probability': '違約機率（DPM模型預測，0-100%）',
                'risk_score': '風險分數（0-100分，越高越安全）',
                'risk_grade': '風險等級（A優良/B中等/C警戒/D危險/E極高風險）',
                'risk_action': '風險管理動作（標準閾值0.5）',
                'risk_action_optimal': f'風險管理動作（最佳閾值{predictor.optimal_threshold:.3f}）',
                'risk_alert': '風險警示（LOW/MEDIUM-HIGH/HIGH/VERY HIGH/CRITICAL）',
                'predicted_default': f'高風險標記-標準（閾值{args.threshold:.3f}）：0=低風險, 1=高風險',
                'predicted_default_optimal': f'高風險標記-最佳（閾值{predictor.optimal_threshold:.3f}）：0=低風險, 1=高風險',
                'threshold_difference': '閾值差異：1=兩種閾值判斷不同, 0=判斷相同'
            }

            calculation_methods = {
                'serial number': '原始資料欄位',
                'application date': '原始資料欄位',
                'applicant': '原始資料欄位',
                'ID': '原始資料欄位',
                'historical_default_rate': '根據政大評分（AMFC2）查表取得。政大評分150-465分，對應違約率100%-4%',
                'default_probability': f'DPM模型預測（Stacking集成：LightGBM+XGBoost+CatBoost）',
                'risk_score': 'risk_score = (1 - default_probability/100) × 100',
                'risk_grade': f'A≥{threshold_score}分, B≥60分, C≥40分, D≥20分, E<20分（動態基於最佳閾值）',
                'risk_action': 'A→正常監控, B→關注名單, C→加強監控, D→立即介入, E→強制催收',
                'risk_action_optimal': f'基於最佳閾值{predictor.optimal_threshold:.3f}調整。當predicted_default_optimal=1時升級管理動作',
                'risk_alert': 'A→LOW RISK, B→MEDIUM-HIGH RISK, C→HIGH RISK, D→VERY HIGH RISK, E→CRITICAL RISK',
                'predicted_default': f'if default_probability >= {args.threshold*100:.1f}%: 1, else: 0',
                'predicted_default_optimal': f'if default_probability >= {predictor.optimal_threshold*100:.1f}%: 1, else: 0（F2-score優化）',
                'threshold_difference': 'if predicted_default != predicted_default_optimal: 1, else: 0'
            }

            desc_df = pd.DataFrame([
                {
                    '欄位名稱': col,
                    '中文說明': field_descriptions.get(col, ''),
                    '計算方式': calculation_methods.get(col, '-')
                }
                for col in result_df_output.columns
            ])
            desc_df.to_excel(writer, sheet_name='欄位說明', index=False)

            # Sheet 3: 風險等級說明（使用前面計算好的 optimal_threshold_pct 和 threshold_score）
            risk_info = pd.DataFrame([
                {
                    '風險等級': 'A (優良)',
                    'Risk Score範圍': f'{threshold_score}-100分',
                    '違約機率範圍': f'0-{optimal_threshold_pct:.0f}%',
                    'Risk Alert': 'LOW RISK',
                    '風險程度': '極低風險',
                    '業界對照': '相當於FICO 740+或台灣銀行A級',
                    '違約率': f'<{optimal_threshold_pct:.0f}%（低於模型閾值）',
                    '業務建議': '強烈建議核准，優質客戶'
                },
                {
                    '風險等級': 'B (中等)',
                    'Risk Score範圍': f'60-{threshold_score-1}分',
                    '違約機率範圍': f'{optimal_threshold_pct+1:.0f}-40%',
                    'Risk Alert': 'MEDIUM-HIGH RISK',
                    '風險程度': '中高風險',
                    '業界對照': '相當於FICO 580-669或台灣銀行C-D級',
                    '違約率': f'{optimal_threshold_pct+1:.0f}-40%',
                    '業務建議': '建議加強審核，提高利率或要求擔保'
                },
                {
                    '風險等級': 'C (警戒)',
                    'Risk Score範圍': '40-59分',
                    '違約機率範圍': '41-60%',
                    'Risk Alert': 'HIGH RISK',
                    '風險程度': '高風險',
                    '業界對照': '相當於FICO 500-579或台灣銀行D-E級',
                    '違約率': '41-60%',
                    '業務建議': '需高階主管審批，需擔保品或連帶保證人'
                },
                {
                    '風險等級': 'D (不良)',
                    'Risk Score範圍': '20-39分',
                    '違約機率範圍': '61-80%',
                    'Risk Alert': 'VERY HIGH RISK',
                    '風險程度': '極高風險',
                    '業界對照': '相當於FICO 300-499或次級信用',
                    '違約率': '61-80%',
                    '業務建議': '建議拒絕，除非有充足擔保或特殊理由'
                },
                {
                    '風險等級': 'E (危險)',
                    'Risk Score範圍': '0-19分',
                    '違約機率範圍': '81-100%',
                    'Risk Alert': 'CRITICAL RISK',
                    '風險程度': '危急風險',
                    '業界對照': '信用破產等級',
                    '違約率': '>80%（極可能違約）',
                    '業務建議': '強烈建議拒絕，幾乎確定違約'
                },
            ])
            risk_info.to_excel(writer, sheet_name='風險等級說明', index=False)

        print(f"[OK] Simplified predictions saved to {args.output}")
        print(f"     - Sheet 1: 預測結果 ({len(result_df_output)} records, {len(result_df_output.columns)} columns)")
        print(f"     - Sheet 2: 欄位說明")
        print(f"     - Sheet 3: 風險等級說明")
    else:
        # 一般版：只輸出關鍵欄位
        result_df_output.to_excel(args.output, index=False, engine='openpyxl')
        print(f"[OK] Predictions saved to {args.output}")
        print(f"     輸出欄位數: {len(result_df_output.columns)}")

    # =========================================================================
    # CSV 輸出（供資料庫匯入使用）
    # =========================================================================
    # 使用與 Excel 相同的檔名，只改副檔名
    base_path = os.path.splitext(args.output)[0]  # 移除 .xlsx
    csv_output = f"{base_path}.csv"
    readme_output = f"{base_path}_README.txt"

    export_to_csv(result_df_output, csv_output, predictor)
    generate_readme(readme_output, predictor)

    print(f"\n[OK] CSV output saved to {csv_output}")
    print(f"[OK] Documentation saved to {readme_output}")

    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
