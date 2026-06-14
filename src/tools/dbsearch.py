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
QDRANT_URL = os.getenv("QDRANT_URL", "")

# Limites de busca — configuráveis via .env
RETRIEVAL_LIMIT = int(os.getenv("RETRIEVAL_LIMIT", "20"))
PREFETCH_LIMIT = int(os.getenv("PREFETCH_LIMIT", "50"))

# Singletons globais — carregados uma única vez e reutilizados entre chamadas
_qdrant_client = None
_embed_model = None
_sparse_model = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            _qdrant_client = QdrantClient(url=QDRANT_URL)
        else:
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = carregar_modelo()
    return _embed_model

def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model

def hybrid_search(query: str, limit: int = None) -> list[dict]:
    """Busca híbrida (Dense + Sparse + RRF Fusion) no Qdrant."""
    if limit is None:
        limit = RETRIEVAL_LIMIT
        
    if not QDRANT_URL and not os.path.exists(QDRANT_PATH):
        return []

    client = get_qdrant_client()
    model = get_embed_model()
    sparse_model = get_sparse_model()
    
    vetor_busca_denso = embedar_query(model, query)
    vetor_busca_esparso = list(sparse_model.embed([query]))[0]
    
    try:
        resultados = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=vetor_busca_esparso.indices.tolist(), 
                        values=vetor_busca_esparso.values.tolist()
                    ),
                    using="sparse",
                    limit=PREFETCH_LIMIT
                ),
                qmodels.Prefetch(
                    query=vetor_busca_denso,
                    using="dense",
                    limit=PREFETCH_LIMIT
                )
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit
        )
        return [res.payload for res in resultados.points]
    except Exception as e:
        logger.error(f"❌ Erro na busca (collection pode não existir): {e}")
        return []

def check_database(query: str = None):
    if not QDRANT_URL and not os.path.exists(QDRANT_PATH):
        logger.error(f"❌ Banco de dados não encontrado no caminho: {QDRANT_PATH}")
        return

    client = get_qdrant_client()
    
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        count = collection_info.points_count
        logger.info(f"✅ STATUS DO BANCO: Operacional")
        logger.info(f"📊 Coleção: '{COLLECTION_NAME}'")
        logger.info(f"🔢 Total de Vetores (Chunks) Indexados: {count}")
        logger.info(f"⚙️  Prefetch: {PREFETCH_LIMIT} | Retrieval Limit: {RETRIEVAL_LIMIT}")
    except Exception as e:
        logger.error(f"❌ Erro ao acessar a coleção '{COLLECTION_NAME}': {e}")
        return

    if query:
        logger.info(f"\n🔍 Realizando Busca HÍBRIDA (RRF) por: '{query}'")
        resultados = hybrid_search(query, limit=3)
        
        logger.info("\n🏆 TOP 3 RESULTADOS ENCONTRADOS:")
        logger.info("-" * 50)
        for i, payload in enumerate(resultados, 1):
            documento = payload.get('document_id', 'Desconhecido')
            breadcrumb = payload.get('breadcrumb', '')
            texto = payload.get('texto', '')
            
            logger.info(f"{i}º LUGAR")
            logger.info(f"Origem: {documento}")
            if breadcrumb:
                logger.info(f"Sessão: {breadcrumb}")
            logger.info(f"Trecho: {texto[:250]}...\n")

if __name__ == "__main__":
    termo_busca = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    check_database(termo_busca)