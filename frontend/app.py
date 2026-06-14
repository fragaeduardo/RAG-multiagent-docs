"""
Interface do Usuário (Frontend) utilizando Chainlit.
Gerencia a interação via chat, exibição de fontes na barra lateral e 
comunicação com o orquestrador do Agente RAG.
"""

import sys
import asyncio
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
    format_sources, link_inline_sources,
    build_workflow, run_single_node, step_summary, STEP_LABELS,
)

# ── Funções Auxiliares de Interface ─────────────────────────────────────────

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

@cl.set_starters
async def set_starters():
    """Perguntas-exemplo clicáveis exibidas na tela inicial vazia."""
    return [
        cl.Starter(
            label="Trancamento de matrícula",
            message="Como funciona o trancamento de matrícula na UFG e por quanto tempo posso trancar?",
        ),
        cl.Starter(
            label="Jubilamento / desligamento",
            message="Em quais situações um aluno pode ser jubilado ou desligado do curso por excesso de prazo?",
        ),
        cl.Starter(
            label="Cotas e heteroidentificação",
            message="Como funcionam as bancas de heteroidentificação e quais critérios elas usam na avaliação?",
        ),
        cl.Starter(
            label="Frequência mínima",
            message="Qual é a frequência mínima exigida para aprovação em uma disciplina e como ela é calculada?",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message):
    """Processa a mensagem do usuário e dirige o grafo RAG nó-a-nó, exibindo cada agente como um passo real."""
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Por favor, envie uma pergunta ou termo técnico para consulta.").send()
        return

    try:
        # Constrói o grafo e dirige a execução manualmente, um nó por vez,
        # para que cada agente apareça como um cl.Step real (com seu output verdadeiro).
        workflow, RAGState = build_workflow()
        state = RAGState(user_query=query)

        current = workflow.entry_point
        guard = 0  # rede de segurança contra loops inesperados
        while current != workflow.END and guard < 30:
            guard += 1
            ran = current

            label = STEP_LABELS.get(ran, ran)
            if ran == "planner" and getattr(state, "retries", 0) > 0:
                label = f"🧭 Planner — repescagem #{state.retries}"

            async with cl.Step(name=label, type="tool") as step:
                state, current = await run_single_node(workflow, ran, state)
                step.output = step_summary(ran, state)

        answer = state.final_answer
        hits = state.final_chunks or []

        if not hits:
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

        await cl.Message(content=answer, actions=actions).send()

    except Exception as exc:
        await cl.Message(content=f"Falha na Execução: {exc}").send()