# --- build stage: resolve and install serving dependencies only (no training extras) ---
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

# --- runtime stage ---
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser
WORKDIR /app
COPY --from=builder --chown=appuser /app /app
ENV PATH="/app/.venv/bin:$PATH" MODEL_PATH=/app/models/model.joblib
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -sf http://localhost:8000/health || exit 1
# Mount a trained model at runtime:  docker run -v ./models:/app/models -p 8000:8000 credit-risk-api
CMD ["uvicorn", "credit_risk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
