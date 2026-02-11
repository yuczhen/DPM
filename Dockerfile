# ── DPM 違約預測系統 — Hugging Face Spaces (Docker) ──
FROM python:3.10-slim

# 系統依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 建立工作目錄
WORKDIR /app

# 複製依賴清單並安裝
COPY web/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案（模型 + 預測碼 + Web）
COPY Train/models/          /app/Train/models/
COPY Train/feature_engineering.py /app/Train/feature_engineering.py
COPY Prediction/predict.py  /app/Prediction/predict.py
COPY web/                   /app/web/

# 設定環境變數
ENV DJANGO_SETTINGS_MODULE=core.settings
ENV DJANGO_DEBUG=False
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 進入 web 目錄執行 Django 指令
WORKDIR /app/web

# 資料庫遷移 + 靜態檔案收集
RUN python manage.py migrate --run-syncdb && \
    python manage.py collectstatic --noinput

# HF Spaces 規定 Port 7860
EXPOSE 7860

CMD ["gunicorn", "core.wsgi:application", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120"]
