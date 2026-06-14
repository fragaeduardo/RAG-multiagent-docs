import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_experimental.text_splitter import SemanticChunker
from models.embeddings import SentenceTransformerEmbeddingsAdapter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_semantic_splitter = None

def get_semantic_splitter():
    global _semantic_splitter
    if _semantic_splitter is None:
        logger.info("[SEMANTIC CHUNKER] Carregando modelo BGE-M3 na GPU para chunking semântico...")
        embeddings = SentenceTransformerEmbeddingsAdapter()
        # breakpoint_threshold_type="percentile" é o padrão do Langchain para cortes com base em similaridade semântica
        _semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return _semantic_splitter

def apply_semantic_split(text: str) -> list[Document]:
    """
    Estratégia de particionamento baseada em similaridade semântica de embeddings.
    Avalia a distância vetorial entre as frases e quebra onde o assunto muda.
    """
    splitter = get_semantic_splitter()
    return splitter.create_documents([text])
