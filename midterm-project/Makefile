.PHONY: install data train serve test smoke notebook docker-build docker-run k8s-deploy

install:
	uv sync --locked

data:
	uv run python scripts/download_data.py

train:
	uv run python train.py

serve:
	uv run uvicorn predict:app --host 0.0.0.0 --port 9696

test:
	uv run pytest -v

smoke:
	uv run python smoke_test.py --url http://127.0.0.1:9696

notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb

docker-build:
	docker build -t bikepulse-api:latest .

docker-run:
	docker run --rm -p 9696:9696 bikepulse-api:latest

k8s-deploy:
	kubectl apply -k k8s/
	kubectl rollout status deployment/bikepulse-api --timeout=120s

