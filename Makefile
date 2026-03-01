.PHONY: dev test lint format lock

dev:
	streamlit run app.py

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

lock:
	pip-compile requirements.in -o requirements.txt
	pip-compile requirements-dev.in -o requirements-dev.txt
