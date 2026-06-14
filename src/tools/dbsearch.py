import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from models.embeddings import carregar_modelo, embedar_query
from fastembed import SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "ufg_institucional"
QDRANT_PATH = "data/qdrant_db"

def check_database(query: str = None):
    if not os.path.exists(QDRANT_PATH):
        logger.error(f"❌ Banco de dados não encontrado no caminho: {QDRANT_PATH}")
        return

    client = QdrantClient(path=QDRANT_PATH)
    
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        count = collection_info.points_count
        logger.info(f"✅ STATUS DO BANCO: Operacional")
        logger.info(f"📊 Coleção: '{COLLECTION_NAME}'")
        logger.info(f"🔢 Total de Vetores (Chunks) Indexados: {count}")
    except Exception as e:
        logger.error(f"❌ Erro ao acessar a coleção '{COLLECTION_NAME}': {e}")
        return

    if query:
        logger.info(f"\n🔍 Realizando Busca HÍBRIDA (RRF) por: '{query}'")
        model = carregar_modelo()
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        vetor_busca_denso = embedar_query(model, query)
        vetor_busca_esparso = list(sparse_model.embed([query]))[0]
        
        resultados = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=vetor_busca_esparso.indices.tolist(), 
                        values=vetor_busca_esparso.values.tolist()
                    ),
                    using="sparse",
                    limit=20
                ),
                qmodels.Prefetch(
                    query=vetor_busca_denso,
                    using="dense",
                    limit=20
                )
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=3
        )
        
        logger.info("\n🏆 TOP 3 RESULTADOS ENCONTRADOS:")
        logger.info("-" * 50)
        for i, hit in enumerate(resultados.points, 1):
            score = hit.score
            payload = hit.payload
            documento = payload.get('document_id', 'Desconhecido')
            breadcrumb = payload.get('breadcrumb', '')
            texto = payload.get('texto', '')
            
            logger.info(f"{i}º LUGAR (Score de Relevância: {score:.4f})")
            logger.info(f"Origem: {documento}")
            if breadcrumb:
                logger.info(f"Sessão: {breadcrumb}")
            logger.info(f"Trecho: {texto[:250]}...\n")

if __name__ == "__main__":
    # Pega os argumentos do terminal como string de busca
    termo_busca = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    check_database(termo_busca)