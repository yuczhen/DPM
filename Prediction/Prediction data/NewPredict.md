# DPM 違約預測系統 - 新客戶預測使用說明

## 📋 概述

本系統可預測新貸款申請人的違約風險，協助審核決策。

**模型性能：**
- AUC: 0.9545（優秀）
- Recall: 93.06%（成功捕獲 93% 的違約案例）
- 使用最佳閾值：0.160

---

## 📝 準備資料

### 方式一：使用範本檔案

1. 開啟範本：`Source/new_client_template.xlsx`
2. 範本包含 3 個 Sheet：
   - **欄位說明**：每個欄位的詳細說明
   - **填寫範例**：3 筆範例資料
   - **空白範本**：可直接填寫

### 方式二：自行建立 Excel 檔案

需包含以下 **21 個必填欄位**：

#### 1. 個人資訊
| 欄位名稱 | 中文名稱 | 範例值 | 說明 |
|---------|---------|--------|------|
| `education` | 教育程度 | 專科/大學 | 高中(含)以下, 專科/大學, 研究所以上 |
| `month salary` | 月薪 | 45000 | 單位: 元/月 |
| `job tenure` | 工作年資 | 2.5 | 單位: 年，可用小數 |

#### 2. 地址資訊
| 欄位名稱 | 中文名稱 | 範例值 | 說明 |
|---------|---------|--------|------|
| `post code of permanent address` | 戶籍地郵遞區號 | 100 | 例: 100, 110, 220 |
| `post code of residential address` | 居住地郵遞區號 | 100 | 例: 100, 110, 220 |
| `residence status` | 居住狀況 | own | own(自有), rent(租), family(家人) |

#### 3. 職業資訊
| 欄位名稱 | 中文名稱 | 範例值 | 說明 |
|---------|---------|--------|------|
| `main business` | 行業別 | 製造業 | 例: 製造業, 服務業, 金融業 |

#### 4. 貸款資訊
| 欄位名稱 | 中文名稱 | 範例值 | 說明 |
|---------|---------|--------|------|
| `product` | 貸款產品 | Standard_Loan | 例: Product_A, Standard_Loan |
| `loan term` | 貸款期數 | 24 | 單位: 月 |
| `paid installments` | 已繳期數 | 0 | **新客戶填 0** |

#### 5. 財務比率
| 欄位名稱 | 中文名稱 | 範例值 | 說明 |
|---------|---------|--------|------|
| `debt_to_income_ratio` | 債務收入比 (DTI) | 0.35 | 0.35 代表 35% |
| `payment_to_income_ratio` | 還款收入比 | 0.15 | 0.15 代表 15% |

#### 6. 逾期歷史（8個月）
| 欄位名稱 | 中文名稱 | 範例值 | 說明 |
|---------|---------|--------|------|
| `number of overdue before the first month` | 第0月逾期次數 | 0 | **新客戶填 0** |
| `number of overdue in the first half of the first month` | 第1月上半逾期次數 | 0 | **新客戶填 0** |
| `number of overdue in the second half of the first month` | 第1月下半逾期次數 | 0 | **新客戶填 0** |
| `number of overdue in the second month` | 第2月逾期次數 | 0 | **新客戶填 0** |
| `number of overdue in the third month` | 第3月逾期次數 | 0 | **新客戶填 0** |
| `number of overdue in the fourth month` | 第4月逾期次數 | 0 | **新客戶填 0** |
| `number of overdue in the fifth month` | 第5月逾期次數 | 0 | **新客戶填 0** |
| `number of overdue in the sixth month` | 第6月逾期次數 | 0 | **新客戶填 0** |

#### 7. 選填欄位（回頭客）
| 欄位名稱 | 中文名稱 | 範例值 | 說明 |
|---------|---------|--------|------|
| `ID` | 客戶ID | A123456789 | 回頭客請填入，否則留空 |
| `application date` | 申請日期 | 2024-10-28 | 格式: YYYY-MM-DD，選填 |

> **注意：** 如果提供 `ID` 和 `application date`，系統會自動判斷是否為回頭客，並計算歷史違約率。**回頭客若曾違約，再違約率高達 95.65%！**

---

## 🚀 執行預測

### Step 1: 開啟命令提示字元

```bash
cd C:\Users\Shirley\_Code\DPM\Prediction
```

### Step 2: 執行預測命令

```bash
python predict.py --input "新客戶資料.xlsx" --output "預測結果.xlsx" --threshold 0.160
```

**參數說明：**
- `--input`：新客戶資料檔案路徑
- `--output`：預測結果輸出檔案路徑
- `--threshold`：違約判定閾值（建議使用 0.160）

**範例：**
```bash
python predict.py --input "Source/new_clients_2024Q4.xlsx" --output "Result/predictions_2024Q4.xlsx" --threshold 0.160
```

---

## 📊 預測結果說明

預測完成後，輸出檔案會包含以下欄位：

| 欄位名稱 | 說明 |
|---------|------|
| `default_probability` | 違約機率（0-1 之間） |
| `predicted_default` | 違約預測（0=正常, 1=違約） |
| `risk_score` | 風險分數（0-100，分數越高越安全） |
| `risk_grade` | 風險等級（A/B/C/D/E） |
| `decision` | 建議決策（APPROVE=核准, REJECT=拒絕） |

### 風險等級說明

| 等級 | 風險分數 | 說明 |
|------|---------|------|
| **A (Excellent)** | 80-100 | 極低風險，強烈建議核准 |
| **B (Good)** | 60-79 | 低風險，建議核准 |
| **C (Fair)** | 40-59 | 中等風險，需人工審核 |
| **D (Poor)** | 20-39 | 高風險，建議拒絕 |
| **E (High Risk)** | 0-19 | 極高風險，強烈建議拒絕 |

---

## ⚠️ 注意事項

1. **新客戶填寫：**
   - 所有逾期欄位填 0
   - `paid installments` 填 0
   - `ID` 和 `application date` 可留空

2. **回頭客填寫：**
   - 必須提供 `ID` 和 `application date`
   - 逾期欄位填入實際歷史紀錄
   - 系統會自動計算歷史違約率

3. **資料格式：**
   - 支援 Excel (.xlsx) 或 CSV (.csv)
   - 欄位名稱必須完全一致（包含大小寫和空格）
   - 數值欄位不可有文字或特殊符號

4. **閾值選擇：**
   - 0.160（推薦）：高 Recall (93%)，適合風險規避策略
   - 0.500（標準）：平衡 Precision 和 Recall

---

## 📈 模型性能指標

基於 13,352 筆真實 DPM 資料測試結果：

| 指標 | 數值 | 說明 |
|------|------|------|
| **AUC** | 0.9545 | 模型區分能力（越接近 1 越好） |
| **Recall** | 93.06% | 成功捕獲 93% 的違約案例 |
| **Precision** | 43.93% | 預測為違約中，實際違約的比例 |
| **Accuracy** | 80.86% | 整體預測正確率 |

**業務影響：**
- 成功捕獲 1,891/2,032 個違約案例
- 只漏掉 141 個違約（6.94%）
- 誤拒率：21.33%（2,414 個正常客戶被誤判）

---

## 🔧 疑難排解

### 問題 1：找不到欄位
**錯誤訊息：** `KeyError: 'month salary'`

**解決方法：**
- 檢查 Excel 欄位名稱是否完全一致
- 欄位名稱區分大小寫
- 注意空格（例如：`month salary` 中間有空格）

### 問題 2：WoE 編碼錯誤
**錯誤訊息：** `KeyError in WoE encoding`

**解決方法：**
- 檢查類別欄位的值是否符合訓練資料
- 教育程度：只能填 `高中(含)以下`, `專科/大學`, `研究所以上`
- 居住狀況：只能填 `own`, `rent`, `family`

### 問題 3：Unicode 編碼錯誤
**解決方法：**
- 確保檔案儲存為 UTF-8 編碼
- 使用 Excel 另存新檔時選擇 UTF-8 CSV

---

## 📞 聯絡資訊

如有任何問題，請聯絡：
- 資料科學團隊
- Email: [待補充]

---

**最後更新：** 2025-01-05
**模型版本：** Stacking Ensemble (XGBoost + LightGBM + CatBoost)
**最佳閾值：** 0.160
