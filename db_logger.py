# -*- coding: utf-8 -*-
"""
DB記錄模組
用於將實驗結果寫入資料庫，方便後續觀察和分析
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()  # 載入 .env 檔案
    print("✅ 環境變數載入成功 (DB Logger)")
except ImportError:
    print("⚠️ python-dotenv 未安裝，使用系統環境變數")

class ExperimentDBLogger:
    """
    實驗資料庫記錄器
    將模型訓練結果記錄到SQLite資料庫
    """

    def __init__(self, db_path: str = None):
        """
        初始化DB記錄器

        Args:
            db_path (str): 資料庫檔案路徑，如果為None則從環境變數讀取
        """
        if db_path is None:
            # 從環境變數取得資料庫路徑
            db_path = os.getenv('DATABASE_PATH', 'experiments.db')

        self.db_path = db_path
        self.db_backup_path = os.getenv('DATABASE_BACKUP_PATH', 'backups/experiments_backup.db')

        # 確保備份目錄存在
        os.makedirs(os.path.dirname(self.db_backup_path), exist_ok=True)

        self._init_db()

    def backup_database(self) -> bool:
        """
        備份資料庫到指定路徑

        Returns:
            bool: 是否成功備份
        """
        try:
            import shutil

            # 確保備份目錄存在
            backup_dir = os.path.dirname(self.db_backup_path)
            if backup_dir and not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            # 複製資料庫檔案
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, self.db_backup_path)
                print(f"✅ 資料庫備份成功: {self.db_backup_path}")
                return True
            else:
                print(f"❌ 資料庫檔案不存在: {self.db_path}")
                return False

        except Exception as e:
            print(f"❌ 資料庫備份失敗: {e}")
            return False

    def _init_db(self):
        """初始化資料庫和表格"""
        conn = sqlite3.connect(self.db_path)

        # 創建實驗記錄表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL UNIQUE,
                model_name TEXT,
                dataset_version TEXT,
                train_samples INTEGER,
                val_samples INTEGER,
                test_samples INTEGER,
                train_period_start TEXT,
                train_period_end TEXT,
                val_period_start TEXT,
                val_period_end TEXT,
                test_period_start TEXT,
                test_period_end TEXT,
                features_used TEXT,
                metrics TEXT,
                wandb_run_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                notes TEXT
            )
        ''')

        # 創建指標歷史表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')

        # 創建模型性能追蹤表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                model_version TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                auc_roc REAL,
                log_loss REAL,
                evaluated_at TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')

        conn.commit()
        conn.close()
        print(f"✅ 資料庫初始化完成: {self.db_path}")

    def log_experiment(self, experiment_record: Dict[str, Any]) -> bool:
        """
        記錄實驗結果到資料庫

        Args:
            experiment_record (dict): 實驗記錄

        Returns:
            bool: 是否成功記錄
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # 準備資料
            record = {
                'experiment_id': experiment_record.get('experiment_id', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'model_name': experiment_record.get('model_name', 'Unknown'),
                'dataset_version': experiment_record.get('dataset_version', 'v1.0'),
                'train_samples': experiment_record.get('train_samples', 0),
                'val_samples': experiment_record.get('val_samples', 0),
                'test_samples': experiment_record.get('test_samples', 0),
                'train_period_start': experiment_record.get('train_period_start'),
                'train_period_end': experiment_record.get('train_period_end'),
                'val_period_start': experiment_record.get('val_period_start'),
                'val_period_end': experiment_record.get('val_period_end'),
                'test_period_start': experiment_record.get('test_period_start'),
                'test_period_end': experiment_record.get('test_period_end'),
                'features_used': json.dumps(experiment_record.get('features_used', [])),
                'metrics': json.dumps(experiment_record.get('metrics', {})),
                'wandb_run_id': experiment_record.get('wandb_run_id'),
                'created_at': experiment_record.get('created_at', datetime.now().isoformat()),
                'updated_at': experiment_record.get('updated_at', datetime.now().isoformat()),
                'notes': experiment_record.get('notes', '')
            }

            # 插入記錄
            columns = ', '.join(record.keys())
            placeholders = ', '.join(['?' for _ in record.values()])
            values = list(record.values())

            query = f'''
                INSERT OR REPLACE INTO experiments
                ({columns})
                VALUES ({placeholders})
            '''

            conn.execute(query, values)

            # 記錄詳細指標
            if 'metrics' in experiment_record:
                metrics = experiment_record['metrics']
                experiment_id = record['experiment_id']

                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        conn.execute('''
                            INSERT INTO metrics_history
                            (experiment_id, metric_name, metric_value, recorded_at)
                            VALUES (?, ?, ?, ?)
                        ''', (experiment_id, metric_name, float(metric_value), datetime.now().isoformat()))

            conn.commit()
            conn.close()

            print(f"✅ 實驗記錄成功: {record['experiment_id']}")
            return True

        except Exception as e:
            print(f"❌ 記錄實驗失敗: {e}")
            return False

    def log_model_performance(self, experiment_id: str, model_metrics: Dict[str, float]) -> bool:
        """
        記錄模型性能指標

        Args:
            experiment_id (str): 實驗ID
            model_metrics (dict): 模型性能指標

        Returns:
            bool: 是否成功記錄
        """
        try:
            conn = sqlite3.connect(self.db_path)

            record = {
                'experiment_id': experiment_id,
                'model_version': model_metrics.get('model_version', 'v1.0'),
                'accuracy': model_metrics.get('accuracy'),
                'precision': model_metrics.get('precision'),
                'recall': model_metrics.get('recall'),
                'f1_score': model_metrics.get('f1_score'),
                'auc_roc': model_metrics.get('auc_roc'),
                'log_loss': model_metrics.get('log_loss'),
                'evaluated_at': datetime.now().isoformat()
            }

            # 移除None值
            record = {k: v for k, v in record.items() if v is not None}

            columns = ', '.join(record.keys())
            placeholders = ', '.join(['?' for _ in record.values()])
            values = list(record.values())

            query = f'''
                INSERT INTO model_performance
                ({columns})
                VALUES ({placeholders})
            '''

            conn.execute(query, values)
            conn.commit()
            conn.close()

            print(f"✅ 模型性能記錄成功: {experiment_id}")
            return True

        except Exception as e:
            print(f"❌ 記錄模型性能失敗: {e}")
            return False

    def get_experiment_history(self, limit: int = 10) -> pd.DataFrame:
        """
        獲取實驗歷史記錄

        Args:
            limit (int): 記錄數量限制

        Returns:
            pd.DataFrame: 實驗歷史
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = f'''
                SELECT
                    experiment_id,
                    model_name,
                    dataset_version,
                    train_samples + val_samples + test_samples as total_samples,
                    metrics,
                    created_at
                FROM experiments
                ORDER BY created_at DESC
                LIMIT {limit}
            '''

            df = pd.read_sql_query(query, conn)
            conn.close()

            # 解析metrics JSON
            if not df.empty:
                df['metrics'] = df['metrics'].apply(lambda x: json.loads(x) if x else {})

            return df

        except Exception as e:
            print(f"❌ 獲取實驗歷史失敗: {e}")
            return pd.DataFrame()

    def get_performance_trend(self, metric_name: str = 'auc_roc') -> pd.DataFrame:
        """
        獲取性能趨勢

        Args:
            metric_name (str): 指標名稱

        Returns:
            pd.DataFrame: 性能趨勢
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = '''
                SELECT
                    e.experiment_id,
                    e.model_name,
                    e.created_at,
                    mh.metric_value,
                    mh.metric_name
                FROM experiments e
                JOIN metrics_history mh ON e.experiment_id = mh.experiment_id
                WHERE mh.metric_name = ?
                ORDER BY e.created_at
            '''

            df = pd.read_sql_query(query, conn, params=(metric_name,))
            conn.close()

            return df

        except Exception as e:
            print(f"❌ 獲取性能趨勢失敗: {e}")
            return pd.DataFrame()

    def create_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        創建性能儀表板資料

        Returns:
            dict: 儀表板資料
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # 最新實驗指標
            latest_query = '''
                SELECT
                    e.experiment_id,
                    e.model_name,
                    e.created_at,
                    mh.metric_name,
                    mh.metric_value
                FROM experiments e
                JOIN metrics_history mh ON e.experiment_id = mh.experiment_id
                WHERE e.created_at = (
                    SELECT MAX(created_at) FROM experiments
                )
            '''

            latest_metrics = pd.read_sql_query(latest_query, conn)

            # 性能趨勢
            trend_query = '''
                SELECT
                    e.created_at,
                    AVG(mh.metric_value) as avg_auc
                FROM experiments e
                JOIN metrics_history mh ON e.experiment_id = mh.experiment_id
                WHERE mh.metric_name = 'auc_roc'
                GROUP BY e.created_at
                ORDER BY e.created_at
            '''

            performance_trend = pd.read_sql_query(trend_query, conn)

            # 模型比較
            comparison_query = '''
                SELECT
                    model_name,
                    AVG(CAST(mh.metric_value AS FLOAT)) as avg_auc,
                    COUNT(*) as experiment_count
                FROM experiments e
                JOIN metrics_history mh ON e.experiment_id = mh.experiment_id
                WHERE mh.metric_name = 'auc_roc'
                GROUP BY model_name
                ORDER BY avg_auc DESC
            '''

            model_comparison = pd.read_sql_query(comparison_query, conn)

            conn.close()

            return {
                'latest_experiment': {
                    'experiment_id': latest_metrics['experiment_id'].iloc[0] if not latest_metrics.empty else None,
                    'metrics': dict(zip(latest_metrics['metric_name'], latest_metrics['metric_value']))
                },
                'performance_trend': performance_trend.to_dict('records'),
                'model_comparison': model_comparison.to_dict('records'),
                'total_experiments': len(latest_metrics['experiment_id'].unique()) if not latest_metrics.empty else 0
            }

        except Exception as e:
            print(f"❌ 創建儀表板資料失敗: {e}")
            return {}

    def export_experiments_to_csv(self, output_path: str = 'experiments_export.csv') -> bool:
        """
        匯出實驗記錄到CSV

        Args:
            output_path (str): 輸出檔案路徑

        Returns:
            bool: 是否成功匯出
        """
        try:
            df = self.get_experiment_history(limit=1000)

            if not df.empty:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"✅ 實驗記錄匯出成功: {output_path}")
                return True
            else:
                print("⚠️ 沒有實驗記錄可匯出")
                return False

        except Exception as e:
            print(f"❌ 匯出失敗: {e}")
            return False


# 使用範例
if __name__ == "__main__":
    # 初始化DB記錄器
    logger = ExperimentDBLogger()

    # 範例實驗記錄
    example_experiment = {
        'experiment_id': 'demo_experiment_001',
        'model_name': 'XGBoost_Ensemble',
        'dataset_version': 'recent_5_years_v1',
        'train_samples': 80000,
        'val_samples': 17000,
        'test_samples': 17000,
        'train_period_start': '2019-01-01',
        'train_period_end': '2022-12-31',
        'val_period_start': '2023-01-01',
        'val_period_end': '2023-08-31',
        'test_period_start': '2023-09-01',
        'test_period_end': '2023-12-31',
        'features_used': ['age', 'income', 'credit_history', 'payment_behavior'],
        'metrics': {
            'accuracy': 0.892,
            'precision': 0.845,
            'recall': 0.782,
            'f1_score': 0.812,
            'auc_roc': 0.923,
            'log_loss': 0.287
        },
        'wandb_run_id': 'demo_run_123',
        'notes': '時間意識訓練的第一個實驗'
    }

    # 記錄實驗
    logger.log_experiment(example_experiment)

    # 記錄模型性能
    model_metrics = {
        'model_version': 'v1.0',
        'accuracy': 0.892,
        'precision': 0.845,
        'recall': 0.782,
        'f1_score': 0.812,
        'auc_roc': 0.923,
        'log_loss': 0.287
    }
    logger.log_model_performance('demo_experiment_001', model_metrics)

    # 獲取歷史記錄
    history = logger.get_experiment_history(limit=5)
    print("\n📊 實驗歷史:")
    print(history)

    # 匯出到CSV
    logger.export_experiments_to_csv()

        print("\n✅ DB記錄範例完成！")


# =============================================
# Prediction Model 專用 DB Logger
# =============================================

class PredictionDBLogger:
    """
    預測模型專用資料庫記錄器
    用於記錄即時預測結果和模型監控
    """

    def __init__(self, db_path: str = None):
        """
        初始化預測DB記錄器

        Args:
            db_path (str): 資料庫檔案路徑，如果為None則從環境變數讀取
        """
        if db_path is None:
            # 從環境變數取得資料庫路徑
            db_path = os.getenv('DATABASE_PATH', 'experiments.db')

        self.db_path = db_path
        self._init_prediction_tables()

    def _init_prediction_tables(self):
        """初始化預測相關的表格"""
        conn = sqlite3.connect(self.db_path)

        # 創建預測記錄表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                model_version TEXT,
                default_probability REAL,
                risk_category TEXT,
                risk_score INTEGER,
                model_predictions TEXT,  -- JSON格式的多模型預測結果
                features_used TEXT,      -- JSON格式的特徵值
                processing_time REAL,    -- 預測處理時間(秒)
                batch_id TEXT,           -- 批次ID
                created_at TEXT
            )
        ''')

        # 創建模型監控表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TEXT,
                alert_triggered BOOLEAN DEFAULT FALSE,
                alert_message TEXT
            )
        ''')

        # 創建預測統計表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_predictions INTEGER,
                avg_default_probability REAL,
                high_risk_count INTEGER,
                medium_risk_count INTEGER,
                low_risk_count INTEGER,
                processing_errors INTEGER,
                UNIQUE(date)
            )
        ''')

        conn.commit()
        conn.close()
        print(f"✅ 預測資料庫表格初始化完成: {self.db_path}")

    def log_prediction(self, prediction_record: Dict[str, Any]) -> bool:
        """
        記錄單次預測結果

        Args:
            prediction_record (dict): 預測記錄

        Returns:
            bool: 是否成功記錄
        """
        try:
            conn = sqlite3.connect(self.db_path)

            record = {
                'client_id': prediction_record.get('client_id', 'UNKNOWN'),
                'prediction_timestamp': prediction_record.get('prediction_timestamp', datetime.now().isoformat()),
                'model_version': prediction_record.get('model_version', 'v1.0'),
                'default_probability': prediction_record.get('default_probability', 0.0),
                'risk_category': prediction_record.get('risk_category', 'UNKNOWN'),
                'risk_score': prediction_record.get('risk_score', 0),
                'model_predictions': json.dumps(prediction_record.get('model_predictions', {})),
                'features_used': json.dumps(prediction_record.get('features_used', {})),
                'processing_time': prediction_record.get('processing_time', 0.0),
                'batch_id': prediction_record.get('batch_id'),
                'created_at': datetime.now().isoformat()
            }

            columns = ', '.join(record.keys())
            placeholders = ', '.join(['?' for _ in record.values()])
            values = list(record.values())

            query = f'''
                INSERT INTO predictions
                ({columns})
                VALUES ({placeholders})
            '''

            conn.execute(query, values)
            conn.commit()
            conn.close()

            print(f"✅ 預測記錄成功: {record['client_id']}")
            return True

        except Exception as e:
            print(f"❌ 預測記錄失敗: {e}")
            return False

    def log_batch_predictions(self, batch_records: List[Dict[str, Any]]) -> int:
        """
        批次記錄預測結果

        Args:
            batch_records (list): 預測記錄列表

        Returns:
            int: 成功記錄的數量
        """
        successful = 0

        for record in batch_records:
            if self.log_prediction(record):
                successful += 1

        print(f"✅ 批次預測記錄完成: {successful}/{len(batch_records)}")
        return successful

    def log_model_metric(self, model_version: str, metric_name: str, metric_value: float,
                         alert_triggered: bool = False, alert_message: str = None) -> bool:
        """
        記錄模型監控指標

        Args:
            model_version (str): 模型版本
            metric_name (str): 指標名稱
            metric_value (float): 指標值
            alert_triggered (bool): 是否觸發警報
            alert_message (str): 警報訊息

        Returns:
            bool: 是否成功記錄
        """
        try:
            conn = sqlite3.connect(self.db_path)

            record = {
                'model_version': model_version,
                'metric_name': metric_name,
                'metric_value': float(metric_value),
                'recorded_at': datetime.now().isoformat(),
                'alert_triggered': alert_triggered,
                'alert_message': alert_message
            }

            columns = ', '.join(record.keys())
            placeholders = ', '.join(['?' for _ in record.values()])
            values = list(record.values())

            query = f'''
                INSERT INTO model_monitoring
                ({columns})
                VALUES ({placeholders})
            '''

            conn.execute(query, values)
            conn.commit()
            conn.close()

            if alert_triggered:
                print(f"🚨 模型監控警報: {metric_name} = {metric_value}")
                if alert_message:
                    print(f"   訊息: {alert_message}")

            return True

        except Exception as e:
            print(f"❌ 模型監控記錄失敗: {e}")
            return False

    def update_prediction_stats(self, date: str = None) -> bool:
        """
        更新預測統計資料

        Args:
            date (str): 統計日期，如果為None則使用今天

        Returns:
            bool: 是否成功更新
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        try:
            conn = sqlite3.connect(self.db_path)

            # 查詢當天預測統計
            query = '''
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(default_probability) as avg_default_probability,
                    SUM(CASE WHEN risk_category = 'High' THEN 1 ELSE 0 END) as high_risk_count,
                    SUM(CASE WHEN risk_category = 'Medium' THEN 1 ELSE 0 END) as medium_risk_count,
                    SUM(CASE WHEN risk_category = 'Low' THEN 1 ELSE 0 END) as low_risk_count
                FROM predictions
                WHERE DATE(prediction_timestamp) = ?
            '''

            stats = pd.read_sql_query(query, conn, params=(date,))

            if not stats.empty:
                record = {
                    'date': date,
                    'total_predictions': int(stats.iloc[0]['total_predictions']),
                    'avg_default_probability': float(stats.iloc[0]['avg_default_probability']) if stats.iloc[0]['avg_default_probability'] is not None else 0.0,
                    'high_risk_count': int(stats.iloc[0]['high_risk_count']),
                    'medium_risk_count': int(stats.iloc[0]['medium_risk_count']),
                    'low_risk_count': int(stats.iloc[0]['low_risk_count']),
                    'processing_errors': 0  # 可以從其他地方計算
                }

                # 使用 UPSERT 語法
                conn.execute('''
                    INSERT OR REPLACE INTO prediction_stats
                    (date, total_predictions, avg_default_probability, high_risk_count,
                     medium_risk_count, low_risk_count, processing_errors)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', tuple(record.values()))

                conn.commit()
                conn.close()

                print(f"✅ 預測統計更新成功: {date}")
                return True
            else:
                print(f"⚠️ 沒有找到 {date} 的預測資料")
                return False

        except Exception as e:
            print(f"❌ 預測統計更新失敗: {e}")
            return False

    def get_prediction_stats(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        獲取指定期間的預測統計

        Args:
            start_date (str): 開始日期 (YYYY-MM-DD)
            end_date (str): 結束日期 (YYYY-MM-DD)

        Returns:
            pd.DataFrame: 預測統計資料
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = '''
                SELECT * FROM prediction_stats
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            '''

            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()

            return df

        except Exception as e:
            print(f"❌ 獲取預測統計失敗: {e}")
            return pd.DataFrame()

    def get_recent_predictions(self, limit: int = 10) -> pd.DataFrame:
        """
        獲取最近的預測記錄

        Args:
            limit (int): 記錄數量限制

        Returns:
            pd.DataFrame: 最近的預測記錄
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = '''
                SELECT
                    client_id,
                    prediction_timestamp,
                    default_probability,
                    risk_category,
                    risk_score,
                    model_version
                FROM predictions
                ORDER BY prediction_timestamp DESC
                LIMIT ?
            '''

            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()

            return df

        except Exception as e:
            print(f"❌ 獲取最近預測失敗: {e}")
            return pd.DataFrame()


# 使用範例
if __name__ == "__main__":
    # 初始化DB記錄器
    logger = ExperimentDBLogger()

    # 範例實驗記錄
    example_experiment = {
        'experiment_id': 'demo_experiment_001',
        'model_name': 'XGBoost_Ensemble',
        'dataset_version': 'recent_5_years_v1',
        'train_samples': 80000,
        'val_samples': 17000,
        'test_samples': 17000,
        'train_period_start': '2019-01-01',
        'train_period_end': '2022-12-31',
        'val_period_start': '2023-01-01',
        'val_period_end': '2023-08-31',
        'test_period_start': '2023-09-01',
        'test_period_end': '2023-12-31',
        'features_used': ['age', 'income', 'credit_history', 'payment_behavior'],
        'metrics': {
            'accuracy': 0.892,
            'precision': 0.845,
            'recall': 0.782,
            'f1_score': 0.812,
            'auc_roc': 0.923,
            'log_loss': 0.287
        },
        'wandb_run_id': 'demo_run_123',
        'notes': '時間意識訓練的第一個實驗'
    }

    # 記錄實驗
    logger.log_experiment(example_experiment)

    # 備份資料庫
    logger.backup_database()

    # 記錄模型性能
    model_metrics = {
        'model_version': 'v1.0',
        'accuracy': 0.892,
        'precision': 0.845,
        'recall': 0.782,
        'f1_score': 0.812,
        'auc_roc': 0.923,
        'log_loss': 0.287
    }
    logger.log_model_performance('demo_experiment_001', model_metrics)

    # 獲取歷史記錄
    history = logger.get_experiment_history(limit=5)
    print("\n📊 實驗歷史:")
    print(history)

    # 匯出到CSV
    logger.export_experiments_to_csv()

    print("\n✅ DB記錄範例完成！")

    # 預測模型範例
    print("\n=== 預測模型 DB 範例 ===")
    pred_logger = PredictionDBLogger()

    # 範例預測記錄
    prediction_example = {
        'client_id': 'CLIENT_001',
        'prediction_timestamp': datetime.now().isoformat(),
        'model_version': 'v1.0',
        'default_probability': 0.15,
        'risk_category': 'Low',
        'risk_score': 750,
        'model_predictions': {
            'XGBoost': 0.12,
            'LightGBM': 0.18,
            'CatBoost': 0.15
        },
        'features_used': {
            'age': 35,
            'income': 60000,
            'credit_history': 120
        },
        'processing_time': 0.05
    }

    # 記錄預測
    pred_logger.log_prediction(prediction_example)

    # 記錄模型監控指標
    pred_logger.log_model_metric('v1.0', 'daily_predictions', 150)
    pred_logger.log_model_metric('v1.0', 'avg_processing_time', 0.03)

    # 更新統計
    pred_logger.update_prediction_stats()

    # 獲取最近預測
    recent_preds = pred_logger.get_recent_predictions(limit=5)
    print("\n📈 最近預測記錄:")
    print(recent_preds)

    print("\n✅ 預測DB記錄範例完成！")
