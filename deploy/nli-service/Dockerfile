# Inference-only NLI endpoint. Weights are NOT baked in: Hugging Face
# mounts the pinned repository at /repository, so the served checkpoint
# is whatever the endpoint was configured with and cannot drift.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# CPU wheels only. The GPU build is an order of magnitude larger and
# this endpoint is sized for CPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
