"""
Lógica de Formatação e Ponte de Execução do RAG.
Define como os documentos e fontes devem ser apresentados na interface Chainlit 
e gerencia a chamada assíncrona ao orquestrador agêntico.
"""

import os
import re
import sys
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

# ── Execução nó-a-nó (passos REAIS na UI) ───────────────────────────────────
# Rótulos amigáveis de cada nó do grafo, exibidos como cl.Step na interface.
STEP_LABELS = {
    "planner": "🧭 Planner — decompondo a pergunta",
    "retriever": "🔎 Retriever — busca híbrida + reranking",
    "synthesizer": "✍️ Sintetizador — redigindo a resposta",
    "judge": "⚖️ Juiz — auditando contra alucinações",
}


def build_workflow():
    """Constrói o grafo do orquestrador e expõe a classe de estado.
    Retorna (workflow, RAGState) para que a UI possa dirigir a execução passo a passo."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.RAG.orchestrator import create_rag_workflow, RAGState
    return create_rag_workflow(), RAGState


async def run_single_node(workflow, node_name: str, state):
    """Executa UM nó do grafo numa thread (sem bloquear a UI) e devolve (novo_estado, próximo_nó).

    Isso permite que o frontend envolva cada agente num cl.Step real, em vez de uma
    animação por timer desacoplada do progresso de verdade.
    """
    import anyio
    node = workflow.nodes[node_name]
    state = await anyio.to_thread.run_sync(node.execute, state)

    if node_name in workflow.conditional_edges:
        next_node = workflow.conditional_edges[node_name](state)
    else:
        next_node = workflow.edges.get(node_name, workflow.END)
    return state, next_node


def step_summary(node_name: str, state) -> str:
    """Extrai do estado uma descrição real do que o nó acabou de produzir."""
    if node_name == "planner":
        linhas = [f"**{len(state.plan)} subquery(s) planejada(s):**"]
        for i, sq in enumerate(state.plan, 1):
            var = len(sq.get("variations", []))
            linhas.append(f"{i}. {sq.get('subquery', '')}  _(+{var} variações)_")
        return "\n".join(linhas)

    if node_name == "retriever":
        n = len(state.final_chunks)
        top = state.final_chunks[0].get("document_id", "—") if state.final_chunks else "—"
        return f"**{n} trechos** selecionados pelo reranker.\n\nDocumento mais relevante: `{top}`."

    if node_name == "synthesizer":
        texto = (state.final_answer or "").strip()
        preview = texto.replace("\n", " ")
        preview = preview[:200] + "…" if len(preview) > 200 else preview
        return f"Rascunho gerado ({len(texto)} caracteres).\n\n> {preview}"

    if node_name == "judge":
        if state.feedback == "APPROVED":
            return "✅ **Aprovado** — resposta consistente com as fontes recuperadas."
        return f"❌ **Reprovado** (acionando repescagem #{state.retries}).\n\n> {state.feedback}"

    return ""


def format_sources(hits: list[Any], indices: list[int] | None = None) -> str:
    """Formata o detalhamento técnico das fontes para a aba lateral de auditoria."""
    lines = ["---", "**DETALHAMENTO TÉCNICO DAS FONTES (UFG)**", ""]
    for pos, hit in enumerate(hits, 1):
        src_num = indices[pos - 1] if indices else pos
        
        # Como o Reranker devolve dicionários em final_chunks
        titulo = hit.get("document_id", "—")
        texto = hit.get("texto", "").strip()
        score = hit.get("rerank_score", 0.0)
        
        lines.extend([
            f"**FONTE {src_num}** | Score de Similaridade: {score:.4f}",
            f"- Documento Base: {titulo}",
            "",
        ])
        if texto:
            # Exibe apenas os primeiros 1000 caracteres para auditoria rápida
            snippet = texto[:1000] + "…" if len(texto) > 1000 else texto
            # Converte as linhas para blockquote para renderizar o Markdown lindamente no UI
            quoted_snippet = "\n".join([f"> {line}" for line in snippet.split("\n")])
            lines.extend([quoted_snippet, "", "---"])
            
    return "\n".join(lines)

def link_inline_sources(answer: str, num_hits: int) -> str:
    """Destaca as citações de fontes no corpo da resposta utilizando negrito."""
    return re.sub(r'(\[[^\]]*Fonte\s?\d+[^\]]*\])', r'**\1**', answer)