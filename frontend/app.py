"""
Interface do Usuário (Frontend) utilizando Chainlit.
Gerencia a interação via chat, exibição de fontes na barra lateral e 
comunicação com o orquestrador do Agente RAG.
"""

import sys
import asyncio
from contextlib import suppress
import anyio.to_thread
import chainlit as cl

# ── Compatibilidade com Python 3.14+ ───────────────────────────────────────
if sys.version_info >= (3, 14):
    def _run_in_executor_py314_compat(self, executor, func, *args):
        future = self.create_future()
        try:
            future.set_result(func(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    asyncio.BaseEventLoop.run_in_executor = _run_in_executor_py314_compat

    async def _run_sync_py314_compat(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args)

    anyio.to_thread.run_sync = _run_sync_py314_compat

# ── Importações e Dependências ──────────────────────────────────────────────
from rag import (
    EMBED_MODEL, QDRANT_URL, COLLECTION, LLM_PROVIDER,
    format_sources, link_inline_sources,
)

def has_generation_layer():
    return True

# ── Funções Auxiliares de Interface ─────────────────────────────────────────

PROCESSING_STEPS = [
    "Verificando coleção vetorial",
    "Planejando consultas de recuperação",
    "Executando recuperação híbrida",
    "Aplicando ranking e deduplicação",
    "Montando contexto de geração",
    "Gerando resposta com base nas fontes",
    "Validando consistência factual",
    "Formatando resposta e referências",
]


def _processing_content(step_index: int) -> str:
    """Monta uma mensagem de progresso curta, em uma única linha."""
    title = PROCESSING_STEPS[min(step_index, len(PROCESSING_STEPS) - 1)]
    return f"**{title}...**"


async def _keep_processing_status_updated(status: cl.Message, stop_event: asyncio.Event):
    """Atualiza a mensagem de status enquanto o orquestrador síncrono roda em thread."""
    step_index = 0
    status.content = _processing_content(step_index)
    await status.update()

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4)
        except asyncio.TimeoutError:
            step_index += 1
            status.content = _processing_content(step_index)
            await status.update()

def _get_sources_elements(hits, name="Detalhamento das Fontes"):
    """Cria os elementos de texto formatados para exibição na aba lateral do Chainlit."""
    if not hits:
        return []
    return [
        cl.Text(name=name, content=format_sources(hits), display="side")
    ]

@cl.action_callback("ver_detalhamento")
async def on_ver_detalhamento(action):
    """Gatilho acionado pelo botão de ação para abrir a aba de documentos técnicos."""
    hits = cl.user_session.get("last_hits")
    if hits:
        elements = _get_sources_elements(hits)
        await cl.Message(
            content="Abrindo documentos técnicos na aba lateral...", 
            elements=elements
        ).send()
    else:
        await cl.Message(content="Falha: As fontes da consulta não foram localizadas na sessão.").send()
    return "Ação de detalhamento executada."

# ── Ciclo de Vida do Chat ───────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Configura e exibe a mensagem de boas-vindas com o status da infraestrutura."""
    status_geracao = LLM_PROVIDER if has_generation_layer() else "Desabilitada (verifique as chaves de API no .env)"
    
    await cl.Message(
        content=(
            "## 🎓 Assistente de Regulamentos e Normativas (UFG)\n"
            "Projeto desenvolvido para a disciplina de LPP - Linguagens e Paradigmas de Programação.\n\n"
            f"**Banco Vetorial (Qdrant)**: `{QDRANT_URL}`\n"
            f"**Modelo de Embedding**: `{EMBED_MODEL}`\n"
            f"**Provedor de Geração**: `{status_geracao}`\n"
            f"**Coleção Ativa**: `{COLLECTION}`"
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Processa a mensagem do usuário e coordena a execução do agente RAG."""
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Por favor, envie uma pergunta ou termo técnico para consulta.").send()
        return

    status = None
    stop_status = asyncio.Event()
    status_task = None

    try:
        # Exibe status contínuo de processamento para deixar o fluxo RAG transparente.
        status = cl.Message(content=_processing_content(0))
        await status.send()
        status_task = asyncio.create_task(_keep_processing_status_updated(status, stop_status))
        
        # Execução do Orquestrador (Fluxo de Grafo)
        from rag import executar_agente
        state = await executar_agente(query)

        stop_status.set()
        with suppress(Exception):
            await status_task
        
        # Validação de erros de infraestrutura ou banco offline
        if state.get("error"):
            await status.remove()
            await cl.Message(content=f"Erro Técnico: {state['error']}").send()
            return

        answer = state.get("answer")
        hits = state.get("retrieved_docs", [])

        if not hits:
            await status.remove()
            await cl.Message(content="Nenhum fundamento normativo foi encontrado para esta consulta.").send()
            return

        # Vincula as citações no texto e armazena fontes para o botão de detalhamento
        answer = link_inline_sources(answer, len(hits))
        cl.user_session.set("last_hits", hits)

        # Configuração do botão para ver fontes originais
        actions = [
            cl.Action(
                name="ver_detalhamento", 
                value="open", 
                label="Ver Detalhamento das Fontes",
                payload={"target": "sources"}
            )
        ]
        
        await status.remove()
        await cl.Message(content=answer, actions=actions).send()

    except Exception as exc:
        stop_status.set()
        if status_task:
            with suppress(Exception):
                await status_task
        if status:
            with suppress(Exception):
                await status.remove()
        await cl.Message(content=f"Falha na Execução: {exc}").send()