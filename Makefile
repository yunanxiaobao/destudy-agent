PYTHON ?= .venv/bin/python

.PHONY: help install build-data ui cli mcp-http test eval docker-up docker-mcp

help:
	@echo "Targets: install | build-data | ui | cli | mcp-http | test | eval | docker-up | docker-mcp"

install:
	python3.13 -m venv .venv
	.venv/bin/pip install -r requirements.txt

build-data:
	$(PYTHON) -m data_pipeline.ingest_daad
	$(PYTHON) -m data_pipeline.crawl --clean
	$(PYTHON) -m data_pipeline.extract_programs
	$(PYTHON) -m data_pipeline.programs_db
	$(PYTHON) -m data_pipeline.chunk
	$(PYTHON) -m data_pipeline.index

ui:
	.venv/bin/streamlit run app/streamlit_app.py

cli:
	$(PYTHON) -m agent.cli --engine langgraph

mcp-http:
	$(PYTHON) -m mcp_server.server --http

test:
	$(PYTHON) -m pytest -v

eval:
	$(PYTHON) -m evaluation.eval_retrieval --k 5
	$(PYTHON) -m evaluation.eval_e2e --limit 10
	$(PYTHON) -m evaluation.eval_agent --engine react
	$(PYTHON) -m evaluation.eval_agent --engine langgraph

docker-up:
	docker compose up --build -d

docker-mcp:
	docker compose --profile mcp up --build -d
