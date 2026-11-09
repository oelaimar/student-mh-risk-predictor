FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/project \
    PYTHONPATH=/project

WORKDIR /project

COPY requirements-ml.txt requirements-app.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.token="${JUPYTER_TOKEN:-jury}" --notebook-dir=/project