# 特徵工程與自動調參使用指南

**更新日期**: 2025-10-09

---

## 一、回答您的問題

### 問題 1: 負債比的 12,309 筆是交集結果嗎？

**答案**: ✓ **是的**

- **計算邏輯**:
  ```
  NCCU_CRM (82,864 筆)
    ∩ total_customer (透過身份證字號)
    = 13,040 筆同時有 loan_amount 和 salary
    → 成功計算 DTI/PTI: 12,309 筆 (差異 731 筆可能因年薪為 0)
  ```

- **資料來源交集**:
  1. NCCU_CRM_cleaned.xlsx → 客戶基本資料、薪資、逾期狀態
  2. total_customer_20211015.xlsx → 貸款金額、每期應繳金額
  3. 透過「身份證字號」(ID) 欄位進行 left join
  4. 整合率: 15.29%

### 問題 2: DTI/PTI 達上限是否正常？

**答案**: ✓ **完全正常且合理**

- **客戶來源特性**: 銀行無法核貸 → 本質條件較差
- **高負債比**: 平均 DTI = 99.84, PTI = 9.99 (接近上限 100/10)
- **違約率**: 6.58% (相較銀行 1-2% 明顯較高)
- **模型價值**: 正因如此，更需要精準的違約預測模型來控管風險

**統計資訊**:
```
DTI (Debt-to-Income Ratio):
  - 平均: 99.84 (接近上限 100)
  - 中位數: 100.00
  - 範圍: 0.02 ~ 100.00

PTI (Payment-to-Income Ratio):
  - 平均: 9.99 (接近上限 10)
  - 中位數: 10.00
  - 範圍: 0.02 ~ 10.00
```

### 問題 3: 地址特徵與自動調參

**答案**: ✓ **已完整實作**

---

## 二、特徵工程方法 (已整合至 main.py)

### 2.1 多種編碼方法 (不只 WoE)

| 編碼方法 | 適用特徵 | 優點 | 類別 | 已實作 |
|---------|---------|------|------|--------|
| **WoE Encoding** | 高基數分類 (郵遞區號 358 種) | 降維、保留違約率、單調性 | `WoEEncoder` | ✓ |
| **Target Encoding** | 所有分類特徵 | 直接編碼違約率、加入平滑 | `TargetEncoder` | ✓ |
| **Geographic Risk** | 郵遞區號 → 違約風險評等 | 明確風險分數、城市/區域層級 | `GeographicRiskEncoder` | ✓ |
| **Frequency Encoding** | 分類特徵 | 編碼出現頻率 | 可自行添加 | - |
| **One-Hot Encoding** | 低基數 (產品別 6 種) | 樹模型友好 | sklearn | - |

### 2.2 地理風險特徵工程 (新增)

已整合至 main.py 的 `GeographicRiskEncoder` 類別：

```python
class GeographicRiskEncoder:
    """
    計算地理位置的違約風險特徵
    """
    def __init__(self, min_samples=10):
        # 最少樣本數限制 (避免過擬合)
        pass

    def fit(self, X, y):
        # 計算每個郵遞區號的違約率
        # 計算城市層級 (前3碼) 的違約率
        pass

    def transform(self, X):
        # 產生特徵:
        # - res_risk_score: 居住地風險分數 (連續值)
        # - res_risk_level: 風險等級 (low/medium/high/very_high)
        # - perm_risk_score: 戶籍地風險分數
        # - city_risk_score: 城市風險分數
        # - address_stability_risk: 地址穩定性綜合風險
        pass
```

**新增特徵**:
1. `res_risk_score`: 居住地郵遞區號違約率 (連續值 0-1)
2. `res_risk_level`: 風險等級分類 (low/medium/high/very_high)
3. `perm_risk_score`: 戶籍地郵遞區號違約率
4. `city_risk_score`: 城市層級 (前3碼) 違約率
5. `address_stability_risk`: 綜合風險 = (地址不一致 × 0.3) + (居住地風險 × 0.7)

**高風險地區範例** (已從實際資料計算):
```
Top 5 Highest Risk Areas:
  1. 221 (新北板橋): 34.75% 違約率
  2. 241 (新北三重): 33.08% 違約率
  3. 104 (台北松山): 30.92% 違約率
  4. 884 (澎湖白沙): 39.13% 違約率 (樣本少)
  5. 952 (台東蘭嶼): 20.00% 違約率 (樣本少)
```

---

## 三、W&B Sweep 自動調參

### 3.1 使用方法

#### **方法 A: 執行自動調參 (推薦)**

```bash
# 在 train 目錄下執行
cd c:\Users\Shirley\_Code\DPM\train
python main.py --sweep
```

這會啟動 **W&B Sweep**，自動嘗試 30 組不同的超參數組合，找出最佳配置。

#### **方法 B: 一般訓練 (手動參數)**

```bash
python main.py
```

使用預設參數進行訓練。

### 3.2 Sweep 配置

已在 main.py 中配置 `train_with_wandb_sweep()` 函數：

```python
sweep_config = {
    'method': 'bayes',  # 貝葉斯優化 (比 grid/random 更聰明)
    'metric': {
        'name': 'val_auc',
        'goal': 'maximize'  # 目標: 最大化驗證集 AUC
    },
    'parameters': {
        # XGBoost 超參數
        'xgb_n_estimators': {'values': [100, 200, 300, 500]},
        'xgb_max_depth': {'values': [3, 4, 5, 6, 7]},
        'xgb_learning_rate': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 0.3},
        'xgb_subsample': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
        'xgb_colsample_bytree': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
        'xgb_min_child_weight': {'values': [1, 3, 5, 7]},

        # LightGBM 超參數
        'lgb_n_estimators': {'values': [100, 200, 300, 500]},
        'lgb_max_depth': {'values': [3, 4, 5, 6, 7]},
        'lgb_learning_rate': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 0.3},
        'lgb_num_leaves': {'values': [15, 31, 63, 127]},

        # 特徵工程選項
        'use_target_encoding': {'values': [True, False]},  # 是否使用 Target Encoding
        'use_geo_risk': {'values': [True, False]},         # 是否使用地理風險特徵
        'use_smote': {'values': [True, False]},            # 是否使用 SMOTE 過採樣
        'scale_pos_weight': {'distribution': 'uniform', 'min': 10, 'max': 20}  # 類別權重
    }
}
```

### 3.3 自動優化內容

W&B Sweep 會自動測試:

1. **模型超參數**: XGBoost 和 LightGBM 的最佳配置
2. **特徵工程組合**:
   - WoE Encoding (固定使用)
   - Target Encoding (開/關)
   - Geographic Risk Features (開/關)
3. **類別不平衡處理**:
   - SMOTE 過採樣 (開/關)
   - scale_pos_weight (10-20 之間)

### 3.4 查看結果

執行後會看到:
```
W&B Sweep ID: xxx-xxx-xxx
Starting hyperparameter optimization...

Run 1/30: val_auc = 0.8234 ...
Run 2/30: val_auc = 0.8456 ...
...

View results: https://wandb.ai/your-username/DPM-AutoTune/sweeps/xxx
```

在 W&B 網頁可以看到:
- 每組參數的 AUC 表現
- 參數重要性分析
- 最佳參數組合
- 視覺化圖表

---

## 四、特徵工程完整清單

### 4.1 原始特徵

| 類別 | 特徵名稱 | 說明 | 缺失率 |
|------|---------|------|--------|
| 財務能力 | month_salary | 月薪 (萬) | 0% |
| 財務能力 | job_tenure | 年資 (年) | 0% |
| 財務能力 | loan_amount | 貸款金額 | 84.71% |
| 財務能力 | monthly_payment | 每期應繳 | 84.71% |
| 聯絡穩定性 | post_code_residential | 居住地郵遞區號 | 0.1% |
| 聯絡穩定性 | post_code_permanent | 戶籍地郵遞區號 | 0.1% |
| 聯絡穩定性 | residence_status | 居住狀況 | 0% |
| 工作穩定性 | main_business | 主要經營業務 | 0% |
| 教育背景 | education | 教育程度 | 0% |
| 產品資訊 | product | 產品別 | 0% |
| 逾期記錄 | overdue_columns (8個) | 逾期次數 | 0% |
| 貸款資訊 | loan_term, paid_installments | 期數 | 0% |
| 信用評分 | creditScore, defaultProb | CCIS 評分 | 49.45% |

### 4.2 衍生特徵

| 類別 | 特徵名稱 | 計算公式 | 優先級 |
|------|---------|---------|--------|
| **財務能力** | | | |
| | payment_progress_ratio | 已繳期數 / 貸款總期數 | 高 |
| | job_stable | 1 if 年資 >= 1 else 0 | 中 |
| | annual_salary | 月薪 × 12 | 高 |
| | income_level | 分段 <3/3-5/5-8/>8 | 中 |
| | debt_to_income_ratio | 貸款金額 / 年薪 | **高** |
| | payment_to_income_ratio | 每期應繳 / 月薪 | 高 |
| **聯絡穩定性** | | | |
| | address_match | 1 if 戶籍 == 居住 else 0 | 高 |
| | residence_stable | 1 if 自有/配偶/親友 else 0 | 中 |
| **逾期行為** | | | |
| | total_overdue_count | Σ(所有逾期次數) | 高 |
| | has_overdue | 1 if total > 0 else 0 | 高 |

### 4.3 編碼特徵 (由 Encoder 產生)

| 編碼器 | 產生特徵 | 數量 | 說明 |
|-------|---------|------|------|
| **WoEEncoder** | {feature}_WoE | 5 | 郵遞區號、業務、居住、教育、產品 |
| **TargetEncoder** | {feature}_target_enc | 5 | 同上，但使用 mean encoding |
| **GeographicRiskEncoder** | res_risk_score | 1 | 居住地風險 (連續) |
| | res_risk_level | 1 | 居住地風險 (分類) |
| | perm_risk_score | 1 | 戶籍地風險 |
| | city_risk_score | 1 | 城市風險 |
| | address_stability_risk | 1 | 綜合地址穩定性風險 |

**最終特徵數**:
- 基礎: 10 數值 + 5 分類 = 15
- WoE: +5
- Target Encoding (可選): +5
- Geo Risk (可選): +5
- **總計**: 15-30 個特徵 (依 Sweep 配置而定)

---

## 五、使用流程

### Step 1: 確認資料已準備好

```bash
# 檢查檔案是否存在
ls train/source/DPM_final_cleaned.xlsx
```

應該包含 85,274 筆資料，63 個欄位。

### Step 2: 執行自動調參 (首次推薦)

```bash
cd train
python main.py --sweep
```

這會:
1. 自動測試 30 組不同的超參數和特徵組合
2. 找出最佳配置
3. 記錄所有結果到 W&B

**預計時間**: 2-4 小時 (視機器效能)

### Step 3: 查看 W&B 結果

在瀏覽器開啟 W&B 連結，查看:
- 最佳 AUC
- 最佳參數組合
- 哪些特徵工程方法最有效

### Step 4: 使用最佳參數訓練正式模型

根據 Sweep 結果，修改 `train_with_real_data()` 中的參數，然後:

```bash
python main.py
```

---

## 六、特徵重要性預測

根據徵審邏輯和 WoE 分析，預期特徵重要性排序:

| 排名 | 特徵 | 預期 IV/重要性 | 類型 |
|------|------|--------------|------|
| 1 | **creditScore** (CCIS 信用評分) | 高 | 原始 |
| 2 | **res_risk_score** (居住地風險) | 高 | 地理 |
| 3 | **debt_to_income_ratio** (負債比) | 高 | 衍生 |
| 4 | **total_overdue_count** (總逾期次數) | 高 | 衍生 |
| 5 | **has_overdue** (是否有逾期) | 高 | 衍生 |
| 6 | **post_code_WoE** (郵遞區號 WoE) | 中-高 | WoE |
| 7 | **payment_to_income_ratio** (月付款比) | 中 | 衍生 |
| 8 | **main_business_WoE** (業務類別 WoE) | 中 | WoE |
| 9 | **month_salary** (月薪) | 中 | 原始 |
| 10 | **address_match** (地址一致) | 中 | 衍生 |

**注意**: education (教育程度) **不是**徵審重點，優先級較低。

---

## 七、常見問題

### Q1: 為什麼要用多種編碼方法？

**A**: 不同編碼方法有不同優勢:
- **WoE**: 保留單調性，適合評分卡
- **Target Encoding**: 直接編碼違約率，適合樹模型
- **Geographic Risk**: 明確的風險分數，業務可解釋

W&B Sweep 會自動測試哪個組合最好。

### Q2: SMOTE 是否會導致過擬合？

**A**: 有可能。因此我們:
1. 只在訓練集使用 SMOTE
2. 用 W&B Sweep 測試 use_smote=True/False
3. 如果測試集 AUC 下降，就不使用

### Q3: 負債比缺失 85%，要如何處理？

**A**: 目前策略:
1. 樹模型可自動處理缺失值 (XGBoost/LightGBM/CatBoost)
2. 缺失本身可能是信號 (無法取得貸款資料 = 風險？)
3. 可添加 `has_loan_data` 缺失指示器

### Q4: 如何確認地理風險特徵有效？

**A**: 透過 W&B Sweep:
- `use_geo_risk=True` vs `use_geo_risk=False`
- 比較 AUC 差異
- 查看特徵重要性分析

---

## 八、下一步建議

### 立即執行 (確認後)

1. ✅ **執行 W&B Sweep 自動調參**
   ```bash
   cd train
   python main.py --sweep
   ```

2. ✅ **查看結果並選擇最佳配置**

3. ✅ **使用最佳參數訓練正式模型**
   ```bash
   python main.py
   ```

### 後續優化 (可選)

1. **新增更多特徵工程**:
   - 年齡區間 (從出生日期計算)
   - 收入穩定性 (月薪變異係數)
   - 產品風險評分

2. **嘗試其他模型**:
   - CatBoost (可直接處理分類特徵)
   - 神經網路 (Deep Learning)

3. **業務規則結合**:
   - 高風險地區自動拒絕
   - 負債比 > 5 自動降額

---

**準備好開始訓練了嗎？請確認以上內容後回覆。**
