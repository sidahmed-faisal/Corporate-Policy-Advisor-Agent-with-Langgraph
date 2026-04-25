FROM python:3.11-slim

# System deps for PyMuPDF and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model so container startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

# Corpus is NOT baked in — mount at runtime:
#   docker run -v $(pwd)/policy_corpus:/app/policy_corpus ...
VOLUME ["/app/policy_corpus"]

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    LLM_PROVIDER=gemini \
    CORPUS_DIR=/app/policy_corpus \
    METADATA_FILE=/app/policy_corpus/metadata.json \
    PORT=8000

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
