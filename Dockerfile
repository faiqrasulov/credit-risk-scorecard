# Two-stage build.
#   1. "train": installs the package into a virtualenv and fits the scorecard. German Credit is
#      downloaded from OpenML during the build, so the build needs internet access.
#   2. runtime: copies that exact virtualenv plus the fitted artifact; no data, no compilers.
# Training inside the image ties the joblib artifact to the scikit-learn version that serves it
# (pickled scikit-learn models are not guaranteed to load across versions), and the image needs
# no model file from the host.
#
#   docker build -t credit-risk-scorecard .
#   docker run --rm -p 8000:8000 credit-risk-scorecard
#   curl localhost:8000/health

FROM python:3.12-slim AS train
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .
RUN credit-risk train --data-home /build/data --model-path /opt/model/scorecard.joblib

FROM python:3.12-slim
COPY --from=train /opt/venv /opt/venv
COPY --from=train /opt/model /opt/model
ENV PATH=/opt/venv/bin:$PATH \
    CREDIT_RISK_MODEL=/opt/model/scorecard.joblib \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
