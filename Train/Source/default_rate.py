import pandas as pd
import numpy as np

# 讀取違約率表
default_rate_csv = pd.read_csv('train/source/違約率.csv', encoding='utf-8-sig')

# 讀取 Excel
df = pd.read_excel('train/source/DPM_data_cleaned.xlsx')
if df.iloc[0]['AMFC2'] == 'AMFC2\n政大評分':
    df = df.iloc[1:].reset_index(drop=True)

df['default rate'] = pd.to_numeric(df['default rate'], errors='coerce')

# 檢查是否有 AMFC2 評分分數欄位
print("Excel 欄位:", df.columns.tolist())
print("\n是否有 'AMFC2 評分分數':", 'AMFC2 評分分數' in df.columns)
