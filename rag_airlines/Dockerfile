FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY agent ./agent
COPY context ./context
COPY safety ./safety
COPY observability ./observability
COPY backend ./backend
COPY evals ./evals
COPY config.py ./
RUN pip install --no-cache-dir -c requirements.lock .
COPY data ./data
COPY prompts ./prompts
COPY index_state.json ./
RUN useradd --create-home app && mkdir -p .runtime && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
