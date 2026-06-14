.PHONY: start down download parse chunk vetorizador limpar search

# Comando para ser rodado no HOST (sua maquina local)
start:
	docker compose up -d --build
	@echo "Entrando no container..."
	docker compose exec rag_etl_app bash

down:
	docker compose down

# Comandos para serem rodados DENTRO do container
download:
	python src/ETL/ingestor.py

parse:
	python src/ETL/parser.py

vetorizador vetorizar:
	python src/ETL/vetorizador.py

search:
	python src/tools/dbsearch.py

limpar:
	python src/tools/clean_db.py

chunk:
	python src/ETL/chunker.py

test-rag:
	python src/RAG/orchestrator.py "$(query)"
