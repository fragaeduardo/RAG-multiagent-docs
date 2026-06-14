"""
Módulo responsável pelo Reranking via Cross-Encoder para ordenação por relevância dos resultados da busca.
"""

import os
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import logging
import torch
import gc

logger = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Instância global do modelo para otimização de tempo de carregamento
_reranker = None

def load_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Carregando modelo Reranker {RERANKER_MODEL}...")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=2048)
    return _reranker

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Executa o reranking dos textos recuperados utilizando o modelo Cross-Encoder carregado em memória.
    """
    if not chunks:
        return []

    model = load_reranker()
    
    # Construção dos pares de inferência (query, documento)
    pairs = [[query, chunk.get("texto", "")] for chunk in chunks]
    
    # Liberação de memória CUDA preventiva antes da alocação de tensores do Cross-Encoder
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    
    # Inferência do modelo. O parâmetro batch_size=4 é essencial para evitar CUDA Out of Memory (OOM) em GPUs com 6GB VRAM.
    scores = model.predict(pairs, batch_size=4)
    
    # Liberação de memória CUDA subsequente à inferência
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    
    # Associação dos scores computados aos chunks respectivos para ordenação
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        scored_chunks.append({
            "chunk": chunk,
            "rerank_score": float(scores[i])
        })
        
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Log diagnóstico para auditoria dos scores de relevância selecionados
    if scored_chunks:
        logger.info(f"[RERANKER] Pool de {len(scored_chunks)} chunks avaliados. Score máx: {scored_chunks[0]['rerank_score']:.4f} | Score mín (do pool): {scored_chunks[-1]['rerank_score']:.4f}")
        for rank, item in enumerate(scored_chunks[:top_k], 1):
            doc_id = item['chunk'].get('document_id', '?')
            breadcrumb = item['chunk'].get('breadcrumb', '')[:60]
            logger.info(f"  #{rank} [{item['rerank_score']:.4f}] {doc_id} → {breadcrumb}")
    
    # Injeta a nota do Reranker no dicionário para auditoria no Frontend
    top_chunks = []
    for item in scored_chunks[:top_k]:
        chunk_data = item["chunk"].copy()
        chunk_data["rerank_score"] = item["rerank_score"]
        top_chunks.append(chunk_data)
        
    return top_chunks
