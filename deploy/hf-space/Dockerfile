# Hugging Face Space (free CPU: 2 vCPU, 16 GB RAM) - builds the API straight from the GitHub repo.
# After pushing new code to GitHub: Space -> Settings -> "Factory rebuild".
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/TSofi/pharma-rag /app
WORKDIR /app

ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/models MAX_MODELS_IN_MEMORY=2
RUN pip install --no-cache-dir -r requirements.txt
# Both embedding models are baked into the image and run locally (no embedding API, no credits).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
RUN chmod -R a+rwX /app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
