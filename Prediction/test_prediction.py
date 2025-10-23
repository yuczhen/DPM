# -*- coding: utf-8 -*-
"""
測試預測系統
============
快速測試 DPM 預測功能是否正常運作
"""

from predict import DPMPredictor
import pandas as pd

print("=" * 80)
print("Testing DPM Prediction System")
print("=" * 80)

# 1. 初始化預測器
print("\n[Step 1] Initializing Predictor...")
predictor = DPMPredictor()

# 2. 載入測試資料（使用訓練資料的前 10 筆）
print("\n[Step 2] Loading Test Data...")
test_df = pd.read_excel('source/DPM_merged_cleaned.xlsx', engine='openpyxl')
test_df = test_df.head(10)  # 只取前 10 筆測試
print(f"✓ Loaded {len(test_df)} test clients")

# 3. 預測（只要機率）
print("\n[Step 3] Predicting Probabilities...")
probas = predictor.predict(test_df, return_proba=True)
print(f"✓ Predictions:")
for i, p in enumerate(probas, 1):
    print(f"  Client {i}: {p:.4f} ({p*100:.2f}% 違約機率)")

# 4. 預測（詳細資訊）
print("\n[Step 4] Predicting with Details...")
results = predictor.predict_with_details(test_df, threshold=0.5)

# 5. 顯示結果
print("\n[Step 5] Results Summary:")
print("=" * 80)
print(results[['default_probability', 'predicted_default', 'risk_score',
              'risk_grade', 'decision']].to_string())

# 6. 儲存結果
output_file = 'test_predictions.xlsx'
results.to_excel(output_file, index=False, engine='openpyxl')
print("\n" + "=" * 80)
print(f"✅ Test Complete! Results saved to {output_file}")
print("=" * 80)

# 7. 測試不同閾值
print("\n[Step 6] Testing Different Thresholds...")
for threshold in [0.3, 0.5, 0.7]:
    results_t = predictor.predict_with_details(test_df, threshold=threshold)
    reject_rate = (results_t['decision'] == 'REJECT').mean()
    print(f"  Threshold {threshold:.1f}: Rejection Rate = {reject_rate:.1%}")

print("\n✅ All tests passed!")
