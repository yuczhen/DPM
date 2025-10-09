# -*- coding: utf-8 -*-
"""
資料處理配置
包含時間分割、wandb整合、DB寫入配置
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# =============================================================================
# 資料分割配置
# =============================================================================

# 時間範圍設定
TIME_CONFIG = {
    'use_recent_years': 5,  # 使用最近5年的資料
    'train_ratio': 0.7,     # 訓練集比例
    'validation_ratio': 0.15, # 驗證集比例
    'test_ratio': 0.15,     # 測試集比例
    'time_column': '進件日期',  # 時間欄位名稱
}

# 資料品質檢查
DATA_QUALITY_CONFIG = {
    'min_samples_per_year': 1000,  # 每年最少樣本數
    'max_missing_rate': 0.8,       # 最大缺失率
    'outlier_threshold': 3.0,      # 異常值閾值
}

# =============================================================================
# W&B 整合配置
# =============================================================================

WANDB_CONFIG = {
    'project_name': 'dpm_credit_risk',  # W&B 專案名稱
    'experiment_name': 'time_aware_training',  # 實驗名稱
    'tags': ['time_series', 'credit_risk', 'concept_drift'],  # 標籤
    'config': {  # 記錄的配置參數
        'model_type': 'ensemble',
        'feature_set': 'temporal_features',
        'data_version': 'recent_5_years',
        'split_method': 'temporal_split',
    }
}

# =============================================================================
# DB 寫入配置
# =============================================================================

DB_CONFIG = {
    'table_name': 'model_experiments',  # 資料表名稱
    'columns': [  # 記錄的欄位
        'experiment_id',
        'model_name',
        'dataset_version',
        'train_samples',
        'val_samples',
        'test_samples',
        'train_period_start',
        'train_period_end',
        'val_period_start',
        'val_period_end',
        'test_period_start',
        'test_period_end',
        'features_used',
        'metrics',
        'wandb_run_id',
        'created_at',
        'updated_at'
    ]
}

# =============================================================================
# 特徵工程配置
# =============================================================================

FEATURE_CONFIG = {
    'temporal_features': [
        'year', 'month', 'quarter', 'day_of_year',
        'days_since_application', 'months_since_application',
        'age_at_application', 'economic_year_indicator'
    ],
    'categorical_features': [
        '婚姻', '教育程度', '居住狀況', 'Bucket',
        '專案名稱', '產品別', '廠商名稱'
    ],
    'numerical_features': [
        '年資', '月薪', '期數', '逾期天數',
        'Current', 'M0', 'M1-1', 'M1-2', 'M2', 'M3', 'M4', 'M5', 'M6'
    ]
}

# =============================================================================
# 模型評估配置
# =============================================================================

EVALUATION_CONFIG = {
    'metrics': [
        'accuracy', 'precision', 'recall', 'f1_score',
        'auc_roc', 'log_loss', 'brier_score'
    ],
    'time_aware_metrics': True,  # 是否使用時間加權評估
    'drift_detection': True,     # 是否啟用飄移檢測
}

# =============================================================================
# 監控配置
# =============================================================================

MONITORING_CONFIG = {
    'drift_detection_interval': 30,  # 天，飄移檢測間隔
    'performance_threshold': 0.8,    # 性能閾值
    'alert_email': None,             # 警報郵件
    'retrain_threshold': 0.05,       # 重新訓練閾值
}


def get_data_time_range(data_path):
    """
    獲取資料的時間範圍

    Args:
        data_path (str): 資料檔案路徑

    Returns:
        dict: 時間範圍資訊
    """
    df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)
    df['進件日期'] = pd.to_datetime(df['進件日期'], errors='coerce')
    df_valid = df.dropna(subset=['進件日期'])

    return {
        'min_date': df_valid['進件日期'].min(),
        'max_date': df_valid['進件日期'].max(),
        'total_samples': len(df_valid),
        'date_range_days': (df_valid['進件日期'].max() - df_valid['進件日期'].min()).days
    }


def filter_recent_years_data(data_path, years=5):
    """
    篩選最近N年的資料

    Args:
        data_path (str): 資料檔案路徑
        years (int): 年數

    Returns:
        pd.DataFrame: 篩選後的資料
    """
    df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)
    df['進件日期'] = pd.to_datetime(df['進件日期'], errors='coerce')
    df_valid = df.dropna(subset=['進件日期'])

    # 計算最近N年的起始日期
    recent_start_date = df_valid['進件日期'].max() - pd.DateOffset(years=years)

    # 篩選資料
    recent_data = df_valid[df_valid['進件日期'] >= recent_start_date].copy()

    print(f"原始資料筆數: {len(df_valid):,}")
    print(f"近{years}年資料筆數: {len(recent_data):,}")
    print(f"篩選起始日期: {recent_start_date.date()}")
    print(f"篩選結束日期: {df_valid['進件日期'].max().date()}")

    return recent_data


def create_temporal_split(data, train_ratio=0.7, val_ratio=0.15):
    """
    建立時間意識的資料分割

    Args:
        data (pd.DataFrame): 輸入資料
        train_ratio (float): 訓練集比例
        val_ratio (float): 驗證集比例

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # 按時間排序
    data_sorted = data.sort_values('進件日期').reset_index(drop=True)

    n_total = len(data_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_data = data_sorted.iloc[:n_train]
    val_data = data_sorted.iloc[n_train:n_train + n_val]
    test_data = data_sorted.iloc[n_train + n_val:]

    # 輸出分割資訊
    print("=== 資料分割資訊 ===")
    print(f"總資料筆數: {n_total","}")
    print(f"訓練集: {len(train_data)","} ({len(train_data)/n_total*100".1f"}%)")
    print(f"驗證集: {len(val_data)","} ({len(val_data)/n_total*100".1f"}%)")
    print(f"測試集: {len(test_data)","} ({len(test_data)/n_total*100".1f"}%)")

    print("
時間範圍:")
    print(f"訓練集: {train_data['進件日期'].min().date()} 至 {train_data['進件日期'].max().date()}")
    print(f"驗證集: {val_data['進件日期'].min().date()} 至 {val_data['進件日期'].max().date()}")
    print(f"測試集: {test_data['進件日期'].min().date()} 至 {test_data['進件日期'].max().date()}")

    # 計算各集合的違約率
    for name, dataset in [('訓練集', train_data), ('驗證集', val_data), ('測試集', test_data)]:
        default_rate = (dataset['Bucket'].isin(['M1-1', 'M1-2', 'M2', 'M3', 'M4', 'M5', 'M6'])).mean() * 100
        print(f"{name}違約率: {default_rate".2f"}%")

    return train_data, val_data, test_data


def create_experiment_record(experiment_id, model_name, train_data, val_data, test_data, metrics, wandb_run_id):
    """
    建立實驗記錄，用於DB寫入

    Args:
        experiment_id (str): 實驗ID
        model_name (str): 模型名稱
        train_data (pd.DataFrame): 訓練資料
        val_data (pd.DataFrame): 驗證資料
        test_data (pd.DataFrame): 測試資料
        metrics (dict): 評估指標
        wandb_run_id (str): W&B運行ID

    Returns:
        dict: 實驗記錄
    """
    return {
        'experiment_id': experiment_id,
        'model_name': model_name,
        'dataset_version': f'recent_{TIME_CONFIG["use_recent_years"]}_years',
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'train_period_start': train_data['進件日期'].min().date() if len(train_data) > 0 else None,
        'train_period_end': train_data['進件日期'].max().date() if len(train_data) > 0 else None,
        'val_period_start': val_data['進件日期'].min().date() if len(val_data) > 0 else None,
        'val_period_end': val_data['進件日期'].max().date() if len(val_data) > 0 else None,
        'test_period_start': test_data['進件日期'].min().date() if len(test_data) > 0 else None,
        'test_period_end': test_data['進件日期'].max().date() if len(test_data) > 0 else None,
        'features_used': FEATURE_CONFIG['temporal_features'] + FEATURE_CONFIG['categorical_features'] + FEATURE_CONFIG['numerical_features'],
        'metrics': metrics,
        'wandb_run_id': wandb_run_id,
        'created_at': datetime.now(),
        'updated_at': datetime.now()
    }
