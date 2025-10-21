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

        print(f"\n✓ Created {len([c for c in df_enhanced.columns if c not in df.columns])} new overdue pattern features")

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

        # 逾期行為 - Basic
        'total_overdue_count',
        'has_overdue',
        'early_overdue_count',

        # 逾期行為 - Tier 1 Enhanced Patterns
        'late_overdue_count',
        'overdue_trend',
        'overdue_worsening',
        'max_overdue_in_month',
        'overdue_std',
        'overdue_consistency',
        'overdue_frequency',
        'overdue_freq_ratio',
        'recent_overdue_severity',
        'max_consecutive_overdue',

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
    print("  - OverduePatternEncoder (NEW - Tier 1 Improvement)")
