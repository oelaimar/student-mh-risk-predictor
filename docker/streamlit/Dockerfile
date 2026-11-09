FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ml.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY app/ app/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]