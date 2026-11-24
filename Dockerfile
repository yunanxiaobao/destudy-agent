FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# Prefer slim CPU wheel on x86_64; fall back to default PyPI wheel on arm64 Linux
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    || pip install --no-cache-dir torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8765

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
