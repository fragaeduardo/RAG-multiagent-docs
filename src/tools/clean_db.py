import os
import shutil
import logging

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "")
COLLECTION_NAME = "ufg_institucional"

def limpar_banco():
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL)
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"✅ Collection '{COLLECTION_NAME}' deletada do Qdrant Server!")
        except Exception:
            logger.info("⚠️ Collection não existia no servidor.")
    else:
        path = "data/qdrant_db"
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info("✅ Banco de dados vetorial local apagado com sucesso!")
        else:
            logger.info("⚠️ O banco de dados já estava limpo.")

if __name__ == "__main__":
    limpar_banco()
