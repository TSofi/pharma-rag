# Container for the API + frontend. Used by Hugging Face Spaces (Docker SDK).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the first request isn't slow.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY frontend ./frontend
COPY data ./data

# HF Spaces runs containers as a non-root user and expects port 7860.
RUN chmod -R a+rX /app
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
