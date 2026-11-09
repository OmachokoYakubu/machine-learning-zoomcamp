#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p docs/images

export PATH="$HOME/.local/bin:$PATH"
uv run uvicorn predict:app --host 127.0.0.1 --port 19696 >/tmp/bikepulse_ui.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 25); do
  if curl -fsS http://127.0.0.1:19696/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

uv run python smoke_test.py --url http://127.0.0.1:19696 > artifacts/smoke_test_output.json

chromium --headless --no-sandbox --disable-gpu \
  --window-size=1280,860 \
  --virtual-time-budget=3000 \
  --screenshot=docs/images/ui_dispatch_console.png \
  http://127.0.0.1:19696/

chromium --headless --no-sandbox --disable-gpu \
  --window-size=1280,900 \
  --virtual-time-budget=3000 \
  --screenshot=docs/images/swagger_api_docs.png \
  http://127.0.0.1:19696/docs

ls -lh docs/images/ui_dispatch_console.png docs/images/swagger_api_docs.png

