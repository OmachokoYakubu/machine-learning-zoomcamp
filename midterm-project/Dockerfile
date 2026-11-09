FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /code

COPY pyproject.toml uv.lock .python-version README.md ./
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --locked --no-dev

COPY model.py predict.py smoke_test.py ./
COPY artifacts/ ./artifacts/
COPY static/ ./static/

EXPOSE 9696

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9696/health', timeout=3)"

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

