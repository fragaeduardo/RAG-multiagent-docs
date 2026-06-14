import os
import sys
import json
import logging
import uuid
import gc
import torch
from tqdm import tqdm

# Ajuste de path para import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, SparseIndexParams, SparseVector
from models.embeddings import carregar_modelo, embedar_documentos, EMBED_DIM
from fastembed import SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silencia logs de tráfego de rede (HuggingFace e Qdrant) para limpar o terminal
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

COLLECTION_NAME = "ufg_institucional"
QDRANT_PATH = "data/qdrant_db"
QDRANT_URL = os.getenv("QDRANT_URL", "")

def init_qdrant(embed_dim: int) -> QdrantClient:
    """Inicializa e cria a collection no Qdrant se não existir."""
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL)
    else:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        client = QdrantClient(path=QDRANT_PATH)
    
    # Verifica se a coleção já existe
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        logger.info(f"Coleção '{COLLECTION_NAME}' já existe.")
    except Exception:
        logger.info(f"Criando nova coleção HÍBRIDA '{COLLECTION_NAME}' (Denso={embed_dim} + Esparso=BM25)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=embed_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
        )
    return client

def processar_e_vetorizar(input_dir: str):
    """Lê os chunks e envia para o banco vetorial."""
    
    # 🧹 Limpeza agressiva da placa de vídeo antes de carregar a IA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Cache da GPU zerado com sucesso para iniciar a vetorização.")
        
    model = carregar_modelo()
    real_dim = model.get_sentence_embedding_dimension() if model else EMBED_DIM
    client = init_qdrant(real_dim)
    
    logger.info("Carregando modelo Sparse (BM25) via FastEmbed...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    # Controle de estado para permitir que o script pare e recomece (Resume-friendly)
    os.makedirs(QDRANT_PATH, exist_ok=True)
    TRACKER_FILE = os.path.join(QDRANT_PATH, "processed_files.json")
    processed_files = set()
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            processed_files = set(json.load(f))
            
    all_json_files = [f for f in os.listdir(input_dir) if f.endswith('_chunks.json')]
    json_files = [f for f in all_json_files if f not in processed_files]
    
    logger.info(f"Encontrados {len(all_json_files)} arquivos no total. {len(processed_files)} já processados. Faltam {len(json_files)}.")
    
    total_chunks = 0
    
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        if not chunks:
            continue
            
        logger.info(f"[{json_files.index(json_file) + 1}/{len(json_files)}] Vetorizando {json_file} ({len(chunks)} chunks)...")
        
        # Lê do .env ou usa 64 como padrão para acelerar
        BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 64))
        
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Lotes de {BATCH_SIZE}", unit="batch"):
            batch = chunks[i:i + BATCH_SIZE]
            
            # Extrair os textos usando a chave `contextualized_content` (que tem os breadcrumbs e anti-amnésia)
            textos_batch = [c.get("contextualized_content", c.get("original_content", str(c))) for c in batch]
            
            # Gerar embeddings densos
            vetores_densos = embedar_documentos(model, textos_batch)
            # Gerar embeddings esparsos (BM25)
            vetores_esparsos = list(sparse_model.embed(textos_batch))
            
            # Criar os PointStructs para o Qdrant
            points = []
            for j, chunk in enumerate(batch):
                vetor_denso = vetores_densos[j]
                vetor_esparso = vetores_esparsos[j]
                # O ID precisa ser UUID ou Int. Vamos converter o chunk_id numa Hash UUID consistente
                chunk_id = chunk.get("chunk_id", f"{json_file}_{i}_{j}")
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                
                payload = {
                    "document_id": chunk.get("document_id", ""),
                    "chunk_id": chunk_id,
                    "breadcrumb": chunk.get("breadcrumb", ""),
                    "metadata": chunk.get("metadata", {}),
                    "texto": textos_batch[j]
                }
                
                vetor_combinado = {
                    "dense": vetor_denso,
                    "sparse": SparseVector(indices=vetor_esparso.indices.tolist(), values=vetor_esparso.values.tolist())
                }
                points.append(PointStruct(id=point_id, vector=vetor_combinado, payload=payload))
            
            # Salvar no banco
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            # 🧹 Libera os resíduos matemáticos deste batch da VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        total_chunks += len(chunks)
        
        # Marca o arquivo como processado com sucesso para nao refazer os calculos na GPU
        processed_files.add(json_file)
        with open(TRACKER_FILE, 'w') as f:
            json.dump(list(processed_files), f)
        
    logger.info(f"Vetorização concluída! {total_chunks} chunks inseridos na coleção '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    PASTA_CHUNKS = "data/chunks/"
    os.makedirs(PASTA_CHUNKS, exist_ok=True)
    processar_e_vetorizar(PASTA_CHUNKS)