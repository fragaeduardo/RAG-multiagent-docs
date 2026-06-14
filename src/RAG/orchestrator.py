import sys
import os
import logging
from typing import Any, Dict
from pydantic import BaseModel, Field
from typing import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.RAG.planner import decompose_and_plan
from src.RAG.compactador import rerank_chunks
from src.RAG.sintetizador import sintetizar_resposta, formatar_contexto
from src.RAG.llm_as_judge import avaliar_resposta
from src.tools.dbsearch import hybrid_search

# Garante que o diretório de logs exista
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'rag_queries.log')

# Configura o logger para imprimir no terminal E salvar no arquivo físico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True, # <--- ESSENCIAL: Força sobrescrever qualquer config prévia das libs
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)

# Silencia os logs HTTP das bibliotecas de ML que enchem o terminal
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("fastembed").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

MAX_RETRIES_ENV = int(os.getenv("MAX_RETRIES", "2"))
RERANKER_TOP_K_ENV = int(os.getenv("RERANKER_TOP_K", "5"))

class SimpleWorkflow:
    END = "END"
    def __init__(self, state_schema):
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.entry_point = None

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, from_node, to_node):
        self.edges[from_node] = to_node

    def add_conditional_edge(self, from_node, condition_fn):
        self.conditional_edges[from_node] = condition_fn

    def set_entry_point(self, name):
        self.entry_point = name

    def compile(self):
        return self

    def invoke(self, state):
        current_node = self.entry_point
        while current_node != self.END:
            if current_node not in self.nodes:
                break
            node = self.nodes[current_node]
            if hasattr(node, 'execute'):
                state = node.execute(state)
            else:
                state = node(state)
            if current_node in self.conditional_edges:
                current_node = self.conditional_edges[current_node](state)
            elif current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                break
        return state

class RAGState(BaseModel):
    """
    Define o estado global do fluxo RAG.
    Armazena as variáveis que transitam entre os nós do grafo de execução.
    """
    user_query: str
    plan: list = []
    all_raw_chunks: list = []  # <--- NOVA MEMÓRIA DE CHUNKS
    final_chunks: list = []
    compacted_context: str = ""
    final_answer: str = ""
    feedback: str = ""
    retries: int = 0
    max_retries: int = MAX_RETRIES_ENV



class PlannerAgent:
    """Nó responsável pelo planejamento e expansão semântica das buscas."""
    def execute(self, state: RAGState) -> RAGState:
        logger.info(f"\n[PLANNER] Decompondo a query original: '{state.user_query}'")
        state.plan = decompose_and_plan(state.user_query, state.feedback)
        
        total_searches = sum(1 + len(subq.get("variations", [])) for subq in state.plan)
        logger.info(f"[PLANNER] Plano: {len(state.plan)} queries × variações = {total_searches} buscas no Qdrant")
        for i, subq in enumerate(state.plan):
            logger.info(f"  Q{i+1}: '{subq.get('subquery', '')[:80]}' + {len(subq.get('variations', []))} variações")
        return state

class RetrieverAgent:
    """Nó responsável pela busca híbrida no Qdrant e reranking dos resultados."""
    def execute(self, state: RAGState) -> RAGState:
        # Se for um retry, a lista all_raw_chunks já terá os chunks da tentativa anterior!
        novos_chunks_na_rodada = 0
        
        for subq in state.plan:
            subquery_text = subq.get("subquery", "")
            variations = subq.get("variations", [subquery_text])
            
            # Executa a busca pela subquery principal e por suas variações
            for search_term in [subquery_text] + variations:
                chunks = hybrid_search(search_term)
                state.all_raw_chunks.extend(chunks)
                novos_chunks_na_rodada += len(chunks)
        
        # Deduplicação baseada no conteúdo textual de TODOS os chunks acumulados no histórico
        unique_chunks = list({c.get("texto", ""): c for c in state.all_raw_chunks if c.get("texto", "")}.values())
        logger.info(f"[RETRIEVER] +{novos_chunks_na_rodada} chunks adicionados ao pool. Pool total acumulado: {len(unique_chunks)} únicos.")
        
        # Reranking global contra a query original do usuário
        state.final_chunks = rerank_chunks(state.user_query, unique_chunks, top_k=RERANKER_TOP_K_ENV)
        logger.info(f"[RETRIEVER] Top {len(state.final_chunks)} chunks selecionados pelo Reranker")
        return state

class SynthesizerAgent:
    """Nó responsável pela geração da resposta final baseada nos chunks recuperados."""
    def execute(self, state: RAGState) -> RAGState:
        logger.info("[SINTETIZADOR] Analisando documentos recuperados para compor a resposta final...")
        contexto_bruto = formatar_contexto(state.final_chunks)
        
        # Passa o rascunho da rodada anterior se estivermos em um retry
        rascunho = state.final_answer if state.retries > 0 else ""
        feedback = state.feedback if state.retries > 0 else ""
        
        state.final_answer = sintetizar_resposta(state.user_query, contexto_bruto, rascunho, feedback)
        return state

class JudgeAgent:
    """Nó avaliador focado na mitigação de alucinações e controle do fluxo de retentativas."""
    def execute(self, state: RAGState) -> RAGState:
        logger.info("[JUDGE] Avaliando a aderência e coerência da resposta...")
        passou, feedback = avaliar_resposta(state.user_query, state.final_answer)
        
        if passou:
            logger.info("[JUDGE] ✅ Aprovado!")
            state.feedback = "APPROVED"
        else:
            logger.warning(f"[JUDGE] ❌ Reprovado: {feedback}")
            state.feedback = feedback
            state.retries += 1
            
        return state

# Constrói o grafo de execução principal do RAG
def create_rag_workflow() -> SimpleWorkflow:
    workflow = SimpleWorkflow(state_schema=RAGState)
    
    # Registro dos nós do fluxo
    workflow.add_node("planner", PlannerAgent())
    workflow.add_node("retriever", RetrieverAgent())
    workflow.add_node("synthesizer", SynthesizerAgent())
    workflow.add_node("judge", JudgeAgent())
    
    # Definição do fluxo linear de execução
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "judge")
    
    # Roteamento condicional pós-avaliação do Judge
    def route_after_judge(state: RAGState) -> str:
        if state.feedback == "APPROVED" or state.retries >= state.max_retries:
            return SimpleWorkflow.END
        return "planner"
        
    workflow.add_conditional_edge("judge", route_after_judge)
    
    # Definição do nó inicial
    workflow.set_entry_point("planner")
    
    return workflow.compile()

if __name__ == "__main__":
    # Execução via CLI para testes locais
    app = create_rag_workflow()
    
    user_input = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "Quais são as regras de avaliação das bancas de heteroidentificação?"
    
    initial_state = RAGState(
        user_query=user_input,
        max_retries=MAX_RETRIES_ENV
    )
    
    logger.info("\n[WORKFLOW] Iniciando orquestrador...")
    final_state = app.invoke(initial_state)
    
    logger.info("\n================ RESPOSTA FINAL ================\n")
    logger.info(final_state.final_answer)
    logger.info("\n================================================\n")
