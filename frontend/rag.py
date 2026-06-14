"""
Lógica de Formatação e Ponte de Execução do RAG.
Define como os documentos e fontes devem ser apresentados na interface Chainlit 
e gerencia a chamada assíncrona ao orquestrador agêntico.
"""

import re
from typing import Any
import os
import re
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# ── Constantes de Configuração para a UI ───────────────────────────────────
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = "ufg_institucional"
LLM_PROVIDER = os.getenv("LLM_SINTETIZADOR_FINAL", "openai/gpt-5.4-nano")

# ── Execução do Agente ──────────────────────────────────────────────────────

async def executar_agente(query: str):
    """Executa o orquestrador do grafo de forma assíncrona para não bloquear a UI."""
    import anyio
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.RAG.orchestrator import create_rag_workflow, RAGState
    
    app = create_rag_workflow()
    initial_state = RAGState(user_query=query)
    
    final_state = await anyio.to_thread.run_sync(app.invoke, initial_state)
    
    # Mapeando o RAGState para o dicionário esperado pelo app.py
    return {
        "answer": final_state.final_answer,
        "retrieved_docs": final_state.final_chunks,
        "error": None
    }

def format_sources(hits: list[Any], indices: list[int] | None = None) -> str:
    """Formata o detalhamento técnico das fontes para a aba lateral de auditoria."""
    lines = ["---", "**DETALHAMENTO TÉCNICO DAS FONTES (UFG)**", ""]
    for pos, hit in enumerate(hits, 1):
        src_num = indices[pos - 1] if indices else pos
        
        # Como o Reranker devolve dicionários em final_chunks
        titulo = hit.get("document_id", "—")
        arquivo = hit.get("arquivo_origem", "—")
        texto = hit.get("texto", "").strip()
        score = hit.get("rerank_score", 0.0)
        
        lines.extend([
            f"**FONTE {src_num}** | Score: {score:.4f}",
            f"- Arquivo: {arquivo}",
            f"- Documento ID: {titulo}",
            "",
        ])
        if texto:
            # Exibe apenas os primeiros 1000 caracteres para auditoria rápida
            snippet = texto[:1000] + "…" if len(texto) > 1000 else texto
            lines.extend([f"```\n{snippet}\n```", ""])
            
    return "\n".join(lines)

def link_inline_sources(answer: str, num_hits: int) -> str:
    """Destaca as citações de fontes no corpo da resposta utilizando negrito."""
    return re.sub(r'(\[[^\]]*Fonte\s?\d+[^\]]*\])', r'**\1**', answer)