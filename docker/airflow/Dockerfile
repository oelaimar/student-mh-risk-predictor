FROM apache/airflow:2.10.5-python3.11

# Install the ML libraries while keeping the image's Airflow version
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-ml.txt