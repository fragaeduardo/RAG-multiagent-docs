"""
Verifica se o modelo não inventou nada e se respondeu direito a pergunta
"""
from .llm import call_llm

def avaliar_resposta(query: str, resposta_gerada: str) -> tuple[bool, str]:
    """
    Devolve um tuple com booleano de aprovação e o motivo
    """
    system_prompt = """Você é um Juiz rigoroso avaliando um sistema de IA RAG.
Você deve julgar se a RESPOSTA GERADA resolve a intenção da PERGUNTA ORIGINAL do usuário.
- Se a resposta trouxer a informação correta, APROVE.
- Se a pergunta pedir um dado exato (ex: um número fixo de faltas), mas a resposta explicar corretamente a regra real do regulamento (ex: exige 75% de frequência), APROVE.
- Se a resposta for evasiva, confusa, ou disser que 'não encontrou informações', REPROVE.
Retorne a primeira linha APENAS como 'APROVADO' ou 'REPROVADO'.
Nas linhas seguintes, forneça um breve feedback justificando.
"""
    prompt = f"PERGUNTA: {query}\nRESPOSTA GERADA: {resposta_gerada}"
    
    avaliacao = call_llm(prompt, system_prompt, role="judge").strip()
    
    linhas = avaliacao.split("\n")
    veredito = linhas[0].strip().upper()
    feedback = "\n".join(linhas[1:]).strip()
    
    passou = "APROVADO" in veredito
    return passou, feedback
