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
1. DECOMPOSIÇÃO: Se a pergunta do usuário contiver duas ou mais dúvidas distintas, quebre-a em 2 ou mais subqueries independentes.
2. REPESCAGEM (FEEDBACK): Se houver um [FEEDBACK] informando que uma parte da pergunta não foi respondida, SUA ÚNICA MISSÃO é criar subqueries EXCLUSIVAMENTE para a parte que faltou (o alvo da crítica do juiz). Ignore as partes da pergunta que já foram respondidas.
3. OTIMIZAÇÃO TÉCNICA: Reescreva a subquery APENAS com a linguagem formal dos regulamentos acadêmicos.
4. VARIAÇÕES: Crie 1 ou no máximo 2 "variations" (sinônimos técnicos) por subquery.

ATENÇÃO AO LINGUAJAR DA UFG:
- "Faltas" / "Limite de faltas" -> "Frequência mínima", "Assiduidade"
- "Jubilamento" -> "Exclusão do curso"
- "Trancar curso" -> "Trancamento de matrícula"
- "Passar de ano" -> "Progressão", "Aprovação"

Retorne SOMENTE um JSON válido neste formato:
[
  {{"subquery": "query principal otimizada tecnicamente", "variations": ["sinônimo técnico 1"]}}
]
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
