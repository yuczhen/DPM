"""
Feature Engineering Module
特徵工程模組 - 包含所有編碼器和特徵轉換邏輯
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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

        print("\n" + "=" * 70)
        print("Target Encoding")
        print("=" * 70)

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
            print(f"  {feature}: {len(smoothed_mean)} categories encoded")

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


class CustomerHistoryEncoder:
    """
    Customer Historical Behavior Feature Engineering (NEW - Tier 1 Critical)
    從客戶歷史借貸記錄中提取行為特徵
    """

    def __init__(self):
        self.history_map = {}
        self.global_default_rate = None

    def fit(self, df, id_col='ID', date_col='application date', target='Default'):
        """
        訓練歷史編碼器，學習每個客戶的歷史記錄
        """
        print("\n" + "=" * 70)
        print("Customer Historical Behavior Feature Engineering")
        print("=" * 70)

        df_sorted = df.sort_values([id_col, date_col]).reset_index(drop=True)
        self.global_default_rate = df[target].mean()

        # 統計
        total_customers = df[id_col].nunique()
        repeat_customers = df[id_col].value_counts()[df[id_col].value_counts() > 1].count()

        print(f"\nTotal customers: {total_customers}")
        print(f"Repeat customers: {repeat_customers} ({repeat_customers/total_customers*100:.1f}%)")

        # 為每個客戶的每筆貸款計算歷史特徵
        for idx, row in df_sorted.iterrows():
            customer_id = row[id_col]

            # 獲取該客戶在此筆貸款之前的所有記錄
            prev_records = df_sorted[
                (df_sorted[id_col] == customer_id) &
                (df_sorted.index < idx)
            ]

            if len(prev_records) == 0:
                # 新客戶
                history = {
                    'is_new_customer': 1,
                    'historical_default': 0,
                    'prev_overdue_days': 0,
                    'prev_payment_ratio': 0,
                    'cumulative_loans': 0,
                    'cumulative_defaults': 0,
                    'customer_default_rate': 0,
                }
            else:
                # 舊客戶
                last_loan = prev_records.iloc[-1]

                history = {
                    'is_new_customer': 0,
                    'historical_default': int(prev_records[target].sum() > 0),  # 曾經違約過
                    'prev_overdue_days': float(last_loan.get('overdue days', 0)),
                    'prev_payment_ratio': float(last_loan.get('paid installments', 0)) / float(last_loan.get('loan term', 1)) if last_loan.get('loan term', 1) > 0 else 0,
                    'cumulative_loans': len(prev_records),
                    'cumulative_defaults': int(prev_records[target].sum()),
                    'customer_default_rate': float(prev_records[target].mean()),
                }

            self.history_map[idx] = history

        # 統計歷史違約影響
        new_defaults = df_sorted[df_sorted.apply(lambda r: self.history_map.get(r.name, {}).get('is_new_customer', 1) == 1, axis=1)][target].mean()
        old_no_default = df_sorted[df_sorted.apply(lambda r: self.history_map.get(r.name, {}).get('historical_default', 0) == 0 and self.history_map.get(r.name, {}).get('is_new_customer', 0) == 0, axis=1)][target].mean() if len(df_sorted[df_sorted.apply(lambda r: self.history_map.get(r.name, {}).get('historical_default', 0) == 0 and self.history_map.get(r.name, {}).get('is_new_customer', 0) == 0, axis=1)]) > 0 else 0
        old_with_default = df_sorted[df_sorted.apply(lambda r: self.history_map.get(r.name, {}).get('historical_default', 0) == 1, axis=1)][target].mean() if len(df_sorted[df_sorted.apply(lambda r: self.history_map.get(r.name, {}).get('historical_default', 0) == 1, axis=1)]) > 0 else 0

        print(f"\nDefault Rate by Customer Type:")
        print(f"  New customers: {new_defaults*100:.2f}%")
        print(f"  Repeat (no prev default): {old_no_default*100:.2f}%")
        print(f"  Repeat (with prev default): {old_with_default*100:.2f}% [HIGH RISK!]")

        print(f"\n[OK] Customer history features created")

    def transform(self, df, id_col='ID', date_col='application date'):
        """
        轉換為歷史特徵
        """
        df_enhanced = df.copy()

        # 如果是新數據（沒有在 fit 時見過），使用新客戶的預設值
        for idx in df_enhanced.index:
            if idx in self.history_map:
                history = self.history_map[idx]
            else:
                # 預測階段的新客戶
                history = {
                    'is_new_customer': 1,
                    'historical_default': 0,
                    'prev_overdue_days': 0,
                    'prev_payment_ratio': 0,
                    'cumulative_loans': 0,
                    'cumulative_defaults': 0,
                    'customer_default_rate': 0,
                }

            for key, value in history.items():
                df_enhanced.at[idx, key] = value

        return df_enhanced

    def fit_transform(self, df, id_col='ID', date_col='application date', target='Default'):
        self.fit(df, id_col, date_col, target)
        return self.transform(df, id_col, date_col)


class OverduePatternEncoder:
    """
    Overdue Pattern Feature Engineering (Tier 1 Improvement)
    從逾期歷史中提取更多行為模式特徵
    """

    def __init__(self):
        self.overdue_cols = [
            'number of overdue before the first month',
            'number of overdue in the first half of the first month',
            'number of overdue in the second half of the first month',
            'number of overdue in the second month',
            'number of overdue in the third month',
            'number of overdue in the fourth month',
            'number of overdue in the fifth month',
            'number of overdue in the sixth month',
        ]

    def create_overdue_pattern_features(self, df):
        """
        創建逾期行為模式特徵 (Tier 1 Improvements)
        """
        print("\n" + "=" * 70)
        print("Overdue Pattern Feature Engineering (Tier 1)")
        print("=" * 70)

        df_enhanced = df.copy()

        # Filter only columns that exist
        existing_overdue_cols = [col for col in self.overdue_cols if col in df.columns]

        if len(existing_overdue_cols) == 0:
            print("[WARNING] No overdue columns found, skipping overdue pattern features")
            return df_enhanced

        print(f"\nFound {len(existing_overdue_cols)} overdue columns")

        # === Feature 1: Early vs Late Overdue Behavior ===
        early_cols = [col for col in existing_overdue_cols if 'first' in col or 'second month' in col or 'before' in col]
        late_cols = [col for col in existing_overdue_cols if 'fourth' in col or 'fifth' in col or 'sixth' in col]

        if early_cols:
            df_enhanced['early_overdue_count'] = df[early_cols].sum(axis=1)
            print("  + early_overdue_count (first 2 months)")

        if late_cols:
            df_enhanced['late_overdue_count'] = df[late_cols].sum(axis=1)
            print("  + late_overdue_count (last 3 months)")

        # === Feature 2: Overdue Trend (Getting worse or better?) ===
        if 'early_overdue_count' in df_enhanced.columns and 'late_overdue_count' in df_enhanced.columns:
            # Positive = getting worse, Negative = improving
            df_enhanced['overdue_trend'] = df_enhanced['late_overdue_count'] - df_enhanced['early_overdue_count']
            df_enhanced['overdue_worsening'] = (df_enhanced['overdue_trend'] > 0).astype(int)
            print("  + overdue_trend (trend indicator)")
            print("  + overdue_worsening (binary: 1=getting worse)")

        # === Feature 3: Maximum Overdue in Any Month ===
        df_enhanced['max_overdue_in_month'] = df[existing_overdue_cols].max(axis=1)
        print("  + max_overdue_in_month (peak overdue)")

        # === Feature 4: Overdue Consistency (Volatility) ===
        df_enhanced['overdue_std'] = df[existing_overdue_cols].std(axis=1).fillna(0)
        df_enhanced['overdue_consistency'] = (df_enhanced['overdue_std'] == 0).astype(int)
        print("  + overdue_std (volatility measure)")
        print("  + overdue_consistency (1=consistent, 0=volatile)")

        # === Feature 5: Overdue Frequency (How many months had overdue?) ===
        df_enhanced['overdue_frequency'] = (df[existing_overdue_cols] > 0).sum(axis=1)
        df_enhanced['overdue_freq_ratio'] = df_enhanced['overdue_frequency'] / len(existing_overdue_cols)
        print("  + overdue_frequency (# of months with overdue)")
        print("  + overdue_freq_ratio (% of months with overdue)")

        # === Feature 6: Recent Overdue Severity (Last 3 months weighted) ===
        if len(late_cols) > 0:
            df_enhanced['recent_overdue_severity'] = df[late_cols].mean(axis=1)
            print("  + recent_overdue_severity (avg overdue in recent months)")

        # === Feature 7: Consecutive Overdue Streak ===
        # Calculate longest consecutive months with overdue
        def calc_max_consecutive_overdue(row):
            """Calculate longest streak of consecutive overdue months"""
            values = [1 if row[col] > 0 else 0 for col in existing_overdue_cols]
            max_streak = 0
            current_streak = 0
            for val in values:
                if val == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            return max_streak

        df_enhanced['max_consecutive_overdue'] = df.apply(calc_max_consecutive_overdue, axis=1)
        print("  + max_consecutive_overdue (longest overdue streak)")

        # === Feature 8: Total Overdue Count ===
        df_enhanced['total_overdue_count'] = df[existing_overdue_cols].sum(axis=1)
        df_enhanced['has_overdue'] = (df_enhanced['total_overdue_count'] > 0).astype(int)
        print("  + total_overdue_count (sum of all overdue)")
        print("  + has_overdue (binary flag)")

        # === Tier 2 Features: Advanced Time Series ===
        print("\n  [Tier 2] Advanced Time Series Features:")

        # Feature 9: Overdue Acceleration (二階導數)
        def calc_acceleration(row):
            """Calculate rate of change in overdue behavior"""
            sequence = [float(row[col]) for col in existing_overdue_cols]
            if len(sequence) < 3:
                return 0
            # Calculate first derivative (velocity)
            velocities = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
            # Calculate second derivative (acceleration)
            accelerations = [velocities[i+1] - velocities[i] for i in range(len(velocities)-1)]
            return np.mean(accelerations) if len(accelerations) > 0 else 0

        df_enhanced['overdue_acceleration'] = df.apply(calc_acceleration, axis=1)
        print("  + overdue_acceleration (rate of change)")

        # Feature 10: Critical Period Overdue (M2-M4，根據分析這是關鍵期)
        critical_cols = [col for col in existing_overdue_cols if 'second half of the first month' in col or 'second month' in col or 'third month' in col or 'fourth month' in col]
        if len(critical_cols) >= 2:
            df_enhanced['critical_period_overdue'] = df[critical_cols].sum(axis=1)
            # Removed: critical_period_severity (redundant - just critical_period_overdue/3)
            print("  + critical_period_overdue (M2-M4 sum)")
            print("  [Removed] critical_period_severity (redundant)")

        # Feature 11: Late Stage Persistence (M4-M7)
        if len(late_cols) >= 2:
            df_enhanced['late_stage_persistence'] = (df[late_cols] > 0).sum(axis=1)
            df_enhanced['late_stage_persistent_flag'] = (df_enhanced['late_stage_persistence'] >= 2).astype(int)
            print("  + late_stage_persistence (# of months with overdue in M4-M7)")
            print("  + late_stage_persistent_flag (1=persistent problem)")

        # Feature 12: Overdue Recovery Rate
        if 'early_overdue_count' in df_enhanced.columns and 'late_overdue_count' in df_enhanced.columns:
            # Positive recovery = early overdue > late overdue (improving)
            df_enhanced['overdue_recovery'] = df_enhanced['early_overdue_count'] - df_enhanced['late_overdue_count']
            df_enhanced['overdue_improving'] = (df_enhanced['overdue_recovery'] > 0).astype(int)
            print("  + overdue_recovery (improvement score)")
            print("  + overdue_improving (1=getting better)")

        # Feature 13: Early Warning Score (M0-M2)
        early_warning_cols = [col for col in existing_overdue_cols if 'before' in col or 'first half of the first month' in col or 'second half of the first month' in col]
        if len(early_warning_cols) >= 2:
            df_enhanced['early_warning_score'] = df[early_warning_cols].sum(axis=1)
            df_enhanced['early_warning_flag'] = (df_enhanced['early_warning_score'] >= 3).astype(int)
            print("  + early_warning_score (M0-M2 sum)")
            print("  + early_warning_flag (1=early warning)")

        # Feature 14: Rolling 3-month Average
        def calc_rolling_avg(row, window=3):
            """Calculate rolling average of overdue"""
            sequence = [float(row[col]) for col in existing_overdue_cols]
            if len(sequence) < window:
                return sequence[-1] if sequence else 0
            # Rolling average for last window months
            return np.mean(sequence[-window:])

        df_enhanced['rolling_3m_overdue_avg'] = df.apply(lambda r: calc_rolling_avg(r, 3), axis=1)
        print("  + rolling_3m_overdue_avg (rolling 3-month avg)")

        # Feature 15: Exponential Weighted Moving Average (給近期更高權重)
        def calc_ema(row, alpha=0.3):
            """Calculate EMA with more weight on recent months"""
            sequence = [float(row[col]) for col in existing_overdue_cols]
            if not sequence:
                return 0
            ema = sequence[0]
            for val in sequence[1:]:
                ema = alpha * val + (1 - alpha) * ema
            return ema

        df_enhanced['ema_overdue'] = df.apply(lambda r: calc_ema(r, 0.3), axis=1)
        print("  + ema_overdue (exponential moving average)")

        print(f"\n[OK] Created {len([c for c in df_enhanced.columns if c not in df.columns])} new overdue pattern features (Tier 1 + Tier 2)")

        return df_enhanced


def get_feature_lists():
    """
    定義特徵清單
    """
    categorical_features = [
        'post code of residential address',
        'main business',
        'residence status',
        'education',
        'product'
    ]

    numerical_features = [
        # 財務能力
        'month salary',
        'job tenure',
        'payment_progress_ratio',
        'job_stable',

        # 聯絡穩定性
        'address_match',
        'residence_stable',

        # DTI (if available)
        'dti_ratio',
        'payment_pressure',

        # 客戶歷史行為 (NEW - Tier 1 Critical)
        'is_new_customer',
        'historical_default',
        'prev_overdue_days',
        'prev_payment_ratio',
        'cumulative_loans',
        'cumulative_defaults',
        'customer_default_rate',

        # 逾期行為 - Basic
        'total_overdue_count',
        'has_overdue',
        'early_overdue_count',

        # 逾期行為 - Tier 1 Enhanced Patterns
        'late_overdue_count',
        'overdue_worsening',
        'max_overdue_in_month',
        'overdue_std',
        'overdue_consistency',
        'overdue_frequency',
        'overdue_freq_ratio',
        'recent_overdue_severity',
        'max_consecutive_overdue',

        # 逾期行為 - Tier 2 Advanced Time Series
        'overdue_acceleration',
        'critical_period_overdue',
        'late_stage_persistence',
        'late_stage_persistent_flag',
        'overdue_recovery',
        'overdue_improving',
        'early_warning_score',
        'early_warning_flag',
        'ema_overdue',

        # 原始特徵
        'loan term',
        'paid installments',
    ]

    # Optional features (may have missing values)
    optional_features = [
        'loan_amount',
        'monthly_payment',
        'debt_to_income_ratio',
        'payment_to_income_ratio',
        'creditScore',
        'defaultProb'
    ]

    return categorical_features, numerical_features, optional_features


if __name__ == "__main__":
    # Test encoders
    print("Feature Engineering Module")
    print("Available encoders:")
    print("  - WoEEncoder")
    print("  - TargetEncoder")
    print("  - GeographicRiskEncoder")
    print("  - CustomerHistoryEncoder (NEW - Tier 1 Critical)")
    print("  - OverduePatternEncoder (Enhanced with Tier 1 + Tier 2)")
