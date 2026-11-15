FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    ROUTER_DB=/app/data/requests.sqlite

WORKDIR /app

COPY requirements-serve.txt .
# CPU-only torch first so sentence-transformers does not pull the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-serve.txt

COPY router/ router/
COPY configs/ configs/
COPY artifacts/ artifacts/
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "router.serve.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
