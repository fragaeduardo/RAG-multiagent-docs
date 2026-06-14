"""
Módulo responsável pela expansão e decomposição semântica das queries de usuário.
"""
import os
import json
import re
from .llm import call_llm

MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", "8"))

def decompose_and_plan(query: str, feedback: str = "") -> list[dict]:
    """
    Decompõe a query de entrada e gera variações semânticas baseadas no vocabulário institucional.
    Retorna uma lista de dicionários no formato:
    [{'subquery': '...', 'variations': ['...', '...']}]
    """

    
    system_prompt = f"""Você é um AI Especialista em Busca (Query Planner) focado nos documentos e regulamentos da UFG.
Sua missão é DECOMPOR perguntas complexas e traduzir termos informais para o jargão técnico.

DIRETRIZES:
1. DECOMPOSIÇÃO: Se a pergunta do usuário contiver subperguntas por dentro, quebre-a em 2 ou mais subqueries para facilitar a busca vetorial independente, mantendo o sentido exato das subperguntas.
2. REPESCAGEM (FEEDBACK): Se houver um [FEEDBACK] informando que uma parte da pergunta não foi respondida, SUA ÚNICA MISSÃO é criar subqueries EXCLUSIVAMENTE para a parte que faltou (o alvo da crítica do juiz). Ignore as partes da pergunta que já foram respondidas.
3. OTIMIZAÇÃO TÉCNICA: Escreva as subqueries sempre em linguagem formal, mas sem exagerar, compatível com documentos da UFG.
4. VARIAÇÕES: Crie 1 ou no máximo 2 variações linguísticas com sinônimos técnicos por subquery.

ATENÇÃO À TRADUÇÃO SEMÂNTICA:
Não se limite a traduções literais. Estudantes costumam usar gírias ou termos informais (ex: "vazar do curso", "rodar de ano", "dp", "matérias"). A sua função é agir como um intérprete institucional: converta sempre a intenção real do aluno para a nomenclatura técnica e genérica encontrada nos estatutos, editais, resoluções e no diário oficial da universidade, mas sem perder o sentido exato do que foi pedido.
IMPORTANTE: Fique extremamente atento a jargões de tempo/época. Se o estudante falar "primeiro período", "calouro" ou "quando eu entrar", traduza para o termo jurídico exato: "semestre de ingresso". 
Cuidado com variações de escrita, siglas e plural/singular que podem impactar a recuperação vetorial.

Retorne SOMENTE um JSON válido. A saída DEVE ser um Array (lista) contendo a quantidade exata de objetos necessários para cobrir todas as dúvidas da pergunta do usuário (seja 1, 3, 5 ou mais). Estrutura obrigatória de cada objeto:
[
  {{"subquery": "texto da subquery otimizada", "variations": ["sinonimo 1", "sinonimo 2"]}}
]
(Gere quantos objetos forem necessários dentro da lista).
"""
    prompt = f"Pergunta do usuário: {query}"
    if feedback:
        prompt += f"\n\n[FEEDBACK DO JUIZ SOBRE A TENTATIVA ANTERIOR]: {feedback}\n\nATENÇÃO: Crie subqueries APENAS para buscar os documentos exigidos pelo Juiz. Não faça queries para as partes da pergunta que ele já considerou respondidas."

    resposta = call_llm(prompt, system_prompt, role="planner")
    
    # Extrai o bloco JSON da resposta, contornando eventuais formatações Markdown do modelo
    try:
        match = re.search(r'\[.*\]', resposta, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(resposta)
    except:
        # Fallback de segurança: retorna a query original em caso de erro no parseamento
        return [{"subquery": query, "variations": [query]}]
