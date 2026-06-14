"""
Módulo de Geração de Embeddings
Gerencia o carregamento de modelos locais (SentenceTransformers) ou uso de APIs 
remotas para transformar textos em vetores densos.
"""

import os
import re
import time
from typing import List

import numpy as np
import torch
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# ── Configurações Globais ──────────────────────────────────────────────────

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "60"))
EMENTA_MAX_CHARS = 400

# ── Processamento de Metadados UFG ─────────────────────────────────────────────

def clean_ufg_filename(filename: str) -> str:
    """Extrai um nome mais legível a partir do nome do arquivo da UFG."""
    if not filename:
        return ""
    # Remove extensões
    nome = re.sub(r"\.(pdf|md|json)$", "", filename, flags=re.IGNORECASE)
    # Remove prefixos técnicos do nosso pipeline
    nome = re.sub(r"^(dados_ufg_|documento_ufg_|akcit_)", "", nome, flags=re.IGNORECASE)
    # Troca hifens e underscores por espaços
    nome = nome.replace("-", " ").replace("_", " ")
    return nome.title().strip()

def build_contexto(meta: dict, secao: str) -> str:
    """Cria um bloco de texto contextual para enriquecer o embedding do chunk."""
    parts = []

    # Nome limpo do documento original
    arquivo_origem = meta.get("arquivo_origem", "")
    if arquivo_origem:
        ident = clean_ufg_filename(arquivo_origem)
        parts.append(f"Documento: {ident}")

    tipo = (meta.get("tipo_documento") or "").strip()
    ementa = (meta.get("ementa") or "").strip()

    if tipo:
        parts.append(f"Tipo: {tipo}")
    if ementa:
        if len(ementa) > EMENTA_MAX_CHARS:
            ementa = ementa[:EMENTA_MAX_CHARS].rsplit(" ", 1)[0] + "..."
        parts.append(f"Resumo/Ementa: {ementa}")
    if secao:
        parts.append(f"Seção: {secao}")

    return "\n".join(parts)

# ── Infraestrutura e Carregamento ──────────────────────────────────────────

def _needs_e5_prefix(model_name: str) -> bool:
    """Verifica se o modelo exige prefixos específicos de query/passage (família E5)."""
    return "e5" in model_name.lower()

def _hf_embed(textos: list[str]) -> list[list[float]]:
    """Gera embeddings utilizando a API remota da Hugging Face."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN não configurado no .env para usar Inference API.")
        
    client = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN, timeout=HF_TIMEOUT)
    for attempt in range(3):
        try:
            result = client.feature_extraction(textos)
            arr = np.array(result, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=1)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return (arr / np.maximum(norms, 1e-9)).tolist()
        except Exception as e:
            if attempt < 2:
                print(f"[EMBEDDINGS] API indisponível, aguardando 20s... (Falha: {e})")
                time.sleep(20)
                continue
            raise RuntimeError(f"HF Embedding API indisponível após 3 tentativas: {e}")

def carregar_modelo() -> SentenceTransformer | None:
    """Carrega o modelo SentenceTransformer na memória (GPU se disponível)."""
    if USE_HF_INFERENCE:
        print(f"[EMBEDDINGS] Modo Inference API remoto: {EMBED_MODEL}")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EMBEDDINGS] Carregando modelo local {EMBED_MODEL} em {device} (fp16)...")
    return SentenceTransformer(
        EMBED_MODEL,
        device=device,
        model_kwargs={"torch_dtype": torch.float16},
        trust_remote_code=True,
    )

# ── Operações de Vetorização ───────────────────────────────────────────────

def embedar_documentos(model: SentenceTransformer, textos: list[str], batch_size: int = 8) -> list[list[float]]:
    """Vetoriza uma lista de textos para o processo de indexação."""
    if _needs_e5_prefix(EMBED_MODEL):
        textos = ["passage: " + t for t in textos]
        
    if model is None:
        return _hf_embed(textos)
        
    # Usa a variável de ambiente se existir, senão usa o padrão (8)
    final_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(batch_size)))
    vecs = model.encode(textos, normalize_embeddings=True, show_progress_bar=False, batch_size=final_batch_size)
    return vecs.tolist()

def embedar_query(model: SentenceTransformer | None, query: str) -> list[float]:
    """Vetoriza uma pergunta para o processo de busca."""
    if _needs_e5_prefix(EMBED_MODEL):
        query = "query: " + query
        
    if model is None:
        return _hf_embed([query])[0]
        
    vec = model.encode(query, normalize_embeddings=True)
    return vec.tolist()

# ── Integração com Frameworks ──────────────────────────────────────────────

class SentenceTransformerEmbeddingsAdapter(Embeddings):
    """Adaptador para compatibilidade do SentenceTransformer com o ecossistema LangChain."""

    def __init__(self, model: SentenceTransformer | None = None) -> None:
        if model is None and not USE_HF_INFERENCE:
            self._model = carregar_modelo()
        else:
            self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embedar_documentos(self._model, texts)

    def embed_query(self, text: str) -> List[float]:
        return embedar_query(self._model, text)
