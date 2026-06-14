import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def limpar_banco():
    path = "data/qdrant_db"
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info("✅ Banco de dados vetorial do Qdrant apagado com sucesso!")
    else:
        logger.info("⚠️ O banco de dados já estava limpo.")

if __name__ == "__main__":
    limpar_banco()
