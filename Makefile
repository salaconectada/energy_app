.PHONY: install run-frontend run-api check test docker-up docker-down

install:
	pip install -r requirements.txt

run-frontend:
	streamlit run app.py

run-api:
	uvicorn backend.app.main:app --reload --port 8000

check:
	python -m compileall app.py frontend backend tests

test:
	python -m unittest discover -s tests -p 'test_*.py' -v

docker-up:
	docker compose up --build

docker-down:
	docker compose down
