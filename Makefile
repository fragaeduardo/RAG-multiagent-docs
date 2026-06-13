.PHONY: start down download parse

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

chunk:
	python src/ETL/chunker.py
