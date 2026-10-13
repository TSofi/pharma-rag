# Container for the API + frontend. Deployed on Render (also works on any Docker host).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    MAX_MODELS_IN_MEMORY=2 \
    FASTEMBED_CACHE_PATH=/app/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Bake int8 versions of both embedding models into the image (~150 MB instead of ~300 MB float16):
# questions are embedded locally, no embedding API needed, and it fits the free 512 MB instance.
ENV INT8_MODEL_DIR=/app/models_int8
RUN python -m app.int8_models
COPY frontend ./frontend
COPY data ./data

RUN chmod -R a+rX /app
# Hosting platforms tell the app which port to use via $PORT (Render: 10000). Default 7860 locally.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
