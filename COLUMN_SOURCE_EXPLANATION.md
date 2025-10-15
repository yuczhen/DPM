# DPM_merged_all 欄位來源說明

**檔案**: DPM_merged_all.xlsx (篩選後 13,352 筆)
**更新日期**: 2025-10-09

---

## AN 欄: default rate (第 40 欄)

### 來源
- **檔案**: `NCCU_CRM_cleaned.xlsx` (原始資料第 42 欄)
- **類型**: 原始資料自帶欄位

### 說明
這是原始 CRM 資料中就存在的欄位，**不是我們計算的**。

可能來源:
1. 歷史違約率統計
2. 某種風險評分系統的輸出
3. 申請時的預測違約率

### 統計資訊
```
資料型態: float64
有效值: 13,352 筆 (100%)
範圍: 4.04% ~ 98.09%
平均: 25.32%
中位數: 10.93%
```

### 範例資料
```
default rate
7.95
13.21
18.53
17.75
7.34
```

### 用途建議
✓ **可作為特徵使用** - 這是歷史風險評估，可能有預測價值
⚠ **注意**: 不要與我們計算的 AY 欄 (Default) 混淆

---

## AS 欄: creditScore (第 45 欄)

### 來源
- **檔案**: `2021CCIS.xlsx`, `2022CCIS.xlsx`, `2023CCIS.xlsx`
- **整合方式**: 透過 `caseNo` (流水號) 與 NCCU_CRM 串接

### 說明
這是 **CCIS 信用評分系統** 產生的信用評分。

CCIS 檔案包含以下欄位:
- `creditScore` - 信用評分 (主要分數)
- `defaultProb` - 違約機率預測
- `scoreLevel` - 評分等級
- 其他子分數: `edu_score`, `gender_score`, `ageTag_score` 等

### 計算方式
CCIS 信用評分是**外部系統計算**的，公式包含:
```
creditScore = f(
    edu_score,           # 教育程度分數
    reasonCount_score,   # 申請原因分數
    gender_score,        # 性別分數
    relatedM2pCnt_score, # 相關 M2+ 次數分數
    empYearTag_score,    # 年資分數
    ageTag_score,        # 年齡分數
    project_score,       # 專案分數
    liveCity_score,      # 居住城市分數
    termTag_score        # 期數分數
)
```

**注意**: 具體公式由 CCIS 系統提供，屬於黑箱模型。

### 統計資訊
```
資料型態: float64
有效值: 3,305 筆 (24.75%)
缺失值: 10,047 筆 (75.25%)
範圍: 269 ~ 427
平均: 358.80
中位數: 359.00
```

### 範例資料
```
creditScore  defaultProb  scoreLevel
361.0        0.1234       B
325.0        0.2567       C
363.0        0.1123       B
335.0        0.2234       C
368.0        0.0987       A
```

### 用途建議
✓ **重要特徵** - 外部信用評分通常有很高的預測力
⚠ **缺失率高** (75.25%) - 需要處理缺失值策略

---

## AY 欄: Default (第 51 欄)

### 來源
- **計算**: 由我們的程式計算產生
- **依據**: E 欄 `overdue status` (逾期狀態)

### 定義: M2+ 規則

**違約定義**: 逾期 >= 60 天視為違約

### 計算公式

```python
def classify_default(overdue_status):
    """
    M2+ 違約定義

    Args:
        overdue_status: 逾期狀態 (Current, M0, M1, M2, M3, ...)

    Returns:
        0 = 正常 (未違約)
        1 = 違約
    """
    status_str = str(overdue_status).strip()

    # 正常: 未逾期或逾期 < 60 天
    if status_str in ['Current', 'M0', 'M1']:
        return 0

    # 違約: 逾期 >= 60 天 (M2+)
    elif status_str.startswith('M'):
        try:
            m_number = int(status_str[1:])  # 提取數字
            return 1 if m_number >= 2 else 0
        except:
            return 0

    return 0
```

### 分類標準

| overdue status | 逾期天數 | Default | 說明 |
|---------------|---------|---------|------|
| **Current** | 0 天 | 0 (正常) | 當期正常 |
| **M0** | 1-30 天 | 0 (正常) | 逾期 < 30 天 |
| **M1** | 31-60 天 | 0 (正常) | 逾期 31-60 天 |
| **M2** | 61-90 天 | **1 (違約)** | 逾期 61-90 天 |
| **M3** | 91-120 天 | **1 (違約)** | 逾期 91-120 天 |
| **M4+** | > 120 天 | **1 (違約)** | 逾期 > 120 天 |

### 統計資訊

```
資料型態: int64 (0 or 1)
有效值: 13,352 筆 (100%)

分佈:
  Default = 0 (正常): 11,320 筆 (84.78%)
  Default = 1 (違約): 2,032 筆 (15.22%)

違約率: 15.22%
```

### 範例資料

```
overdue status  Default  說明
M32             1        違約 (逾期 32 個月)
Current         0        正常
M30             1        違約 (逾期 30 個月)
M36             1        違約 (逾期 36 個月)
M0              0        正常 (逾期 < 30 天)
M1              0        正常 (逾期 31-60 天)
```

### 用途
✓ **目標變數** - 這是我們要預測的 y 值
✓ **訓練標籤** - 用於監督式學習

---

## 三個欄位的關係

### 時間序列
```
申請時 → 撥款後 → 還款中/逾期
   ↓         ↓         ↓
default   creditScore  Default
 rate                  (實際)
(預測)    (信用評分)
```

### 功能對比

| 欄位 | 類型 | 時間點 | 用途 | 缺失率 |
|------|------|-------|------|--------|
| **default rate (AN)** | 預測風險 | 申請時 | 可作為特徵 | 0% |
| **creditScore (AS)** | 信用評分 | 申請時 | 可作為特徵 | 75.25% |
| **Default (AY)** | 實際違約 | 還款後 | **目標變數** | 0% |

### 使用建議

#### 作為特徵 (X)
- ✓ `default rate` - 歷史風險評估
- ✓ `creditScore` - CCIS 信用評分 (需處理缺失)
- ✗ `Default` - 這是目標變數，不能作為特徵！

#### 作為目標 (y)
- ✗ `default rate` - 這是預測值，不是實際值
- ✗ `creditScore` - 這是評分，不是違約標籤
- ✓ `Default` - **這才是我們要預測的目標**

---

## 重要提醒

### ⚠ 不要混淆!

**default rate (AN 欄)** ≠ **Default (AY 欄)**

- `default rate`: 申請時的**預測風險** (0-100%)
- `Default`: 實際還款後的**違約結果** (0 或 1)

### 建模時

```python
# 正確的做法
X = df[['default rate', 'creditScore', 'month salary', ...]]  # 特徵
y = df['Default']  # 目標變數 (0 or 1)

# ❌ 錯誤! 不要用 default rate 作為目標
y = df['default rate']  # 這是預測值，不是真實標籤
```

### 驗證邏輯

如果 `default rate` (預測) 與 `Default` (實際) 完全相關，那原始系統就已經很準確了。我們的任務是:
1. 訓練新模型預測 `Default`
2. 可以使用 `default rate` 作為輔助特徵
3. 比較我們的模型是否比原始 `default rate` 更準確

---

## 查看原始資料

如果您想確認這些欄位的原始來源:

### default rate
```bash
# 查看 NCCU_CRM_cleaned.xlsx 第 42 欄
# Excel 中直接開啟檢視
```

### creditScore
```bash
# 查看 2021CCIS.xlsx, 2022CCIS.xlsx, 2023CCIS.xlsx
# 這些檔案包含 creditScore, defaultProb, scoreLevel 等欄位
```

### Default
```bash
# 這是程式計算的，查看 filter_data.py 第 39-47 行
# 或任何包含 classify_default() 函數的程式
```

---

**總結**:
- AN 欄 (default rate) = 原始資料自帶的風險預測值
- AS 欄 (creditScore) = CCIS 系統計算的信用評分
- AY 欄 (Default) = 我們根據 M2+ 規則計算的實際違約標籤