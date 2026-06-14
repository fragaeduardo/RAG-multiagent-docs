"""
Módulo de síntese responsável por agregar o contexto recuperado e instanciar a geração da resposta final.
"""
from .llm import call_llm

def formatar_contexto(chunks_relevantes: list[dict]) -> str:
    """
    Concatena os chunks filtrados pelo reranker sem sumarização intermediária via LLM.
    Assegura a preservação integral das métricas, datas e artigos originais para a síntese.
    """
    if not chunks_relevantes:
        return ""
    
    textos = []
    for i, chunk in enumerate(chunks_relevantes, 1):
        doc_id = chunk.get("document_id", "Desconhecido")
        breadcrumb = chunk.get("breadcrumb", "")
        texto = chunk.get("texto", "")
        
        cabecalho = f"--- [FONTE {i}: {doc_id}"
        if breadcrumb:
            cabecalho += f" | {breadcrumb}"
        cabecalho += "] ---"
        
        textos.append(f"{cabecalho}\n{texto}")
        
    return "\n\n".join(textos)

def sintetizar_resposta(query: str, contexto_formatado: str, rascunho_anterior: str = "", feedback_juiz: str = "") -> str:
    """
    Realiza a geração da resposta definitiva através da análise sintática e semântica do contexto formatado.
    Permite complementar um rascunho anterior caso o juiz tenha reprovado por falta de completude.
    """
    if not contexto_formatado:
        return "Desculpe, não encontrei nenhuma diretriz ou documento na base de dados da UFG que aborde este tema de forma clara."
        
    system_prompt = """Você é o Assistente Especialista Institucional da UFG.
Sua missão é responder à pergunta do usuário baseando-se EXCLUSIVAMENTE nos [DOCUMENTOS RECUPERADOS] fornecidos.

DIRETRIZES DE OURO:
1. PRECISÃO ABSOLUTA: Se os documentos não tiverem a resposta e não tiverem nada a ver com a pergunta, não enrole. Diga de forma curta e direta que a informação não consta na base de dados e pronto. Só ofereça alguma pequena explicação extra se os documentos tiverem algo minimamente relacionado.
2. CITAÇÕES: Sempre que mencionar um prazo, regra ou artigo, referencie a fonte (ex: "Segundo o Regulamento Geral...").
3. DIDÁTICA: Explique como se estivesse orientando um aluno de forma clara, mas use o rigor técnico de um servidor público.
4. LÓGICA DE CÁLCULO: Se o aluno pedir "número de faltas", mas o documento fala em "75% de frequência mínima", faça a matemática ou explique a regra da porcentagem de forma impecável.
5. FORMATAÇÃO: Use negrito para destacar prazos, termos-chave e use bullet points para listas.
6. ZERO INTERAÇÃO: JAMAIS termine a resposta fazendo perguntas ao usuário, oferecendo ajuda futura, ou pedindo mais dados (ex: "se você me disser seu curso, eu calculo"). A sua resposta deve ser terminativa, oficial e unilateral, atuando como um oráculo de consulta.
"""
    
    if rascunho_anterior and feedback_juiz:
        prompt = f"""[DOCUMENTOS RECUPERADOS ATUALIZADOS]
{contexto_formatado}

[SEU RASCUNHO ANTERIOR]
{rascunho_anterior}

[CRÍTICA DO JUIZ SOBRE O SEU RASCUNHO]
{feedback_juiz}

[PERGUNTA DO USUÁRIO]
{query}

[INSTRUÇÃO DE CORREÇÃO]
Não jogue fora o que estava certo no seu rascunho anterior! Apenas MANTENHA o que já respondia a pergunta e COMPLETE/CORRIJA a parte que faltou (citada pelo Juiz) usando os novos documentos recuperados.

[RESPOSTA DEFINITIVA E EMBASADA]:"""
    else:
        prompt = f"[DOCUMENTOS RECUPERADOS da Base de Dados da UFG]\n{contexto_formatado}\n\n[PERGUNTA DO USUÁRIO]\n{query}\n\n[RESPOSTA DEFINITIVA E EMBASADA]:"
    
    return call_llm(prompt, system_prompt, role="sintetizador")
