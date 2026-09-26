FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=/app
WORKDIR /app
COPY requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
RUN pip install --no-cache-dir greenlet==3.5.6
COPY steel_energy ./steel_energy
COPY config ./config
COPY data ./data
COPY tests ./tests
COPY scripts ./scripts
COPY dags ./dags
COPY report ./report
CMD ["python", "-m", "steel_energy.cli", "all", "--confirm-final", "--sensitivity"]

FROM app AS airflow
COPY constraints-airflow.txt /tmp/constraints-airflow.txt
RUN python -m venv /opt/airflow-venv && \
    /opt/airflow-venv/bin/pip install --no-cache-dir apache-airflow==2.10.5 \
      --constraint /tmp/constraints-airflow.txt
ENV AIRFLOW_HOME=/var/lib/airflow \
    PATH=/opt/airflow-venv/bin:$PATH \
    AIRFLOW__CORE__DAGS_FOLDER=/app/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=false \
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true \
    AIRFLOW__CORE__EXECUTOR=SequentialExecutor \
    CORE_PYTHON=/usr/local/bin/python
CMD ["/opt/airflow-venv/bin/airflow", "standalone"]
