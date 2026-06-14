import os
from openai import OpenAI
from google import genai
import logging

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Mapeamento dos modelos baseado na tarefa
MODEL_PLANNER = os.getenv("LLM_PLANNER", "openai/gpt-4o")
MODEL_SINTETIZADOR = os.getenv("LLM_SINTETIZADOR_FINAL", "openai/gpt-4o")
MODEL_JUDGE = os.getenv("LLM_JUDGE", "anthropic/claude-3.5-sonnet")
MODEL_RESUMIDOR = os.getenv("LLM_RESUMIDOR_COMPACTADOR", "qwen/qwen-2.5-7b-instruct")

def get_client():
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "coloque_sua_chave_aqui":
        raise ValueError("OPENROUTER_API_KEY não está configurada no .env!")
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

def call_llm(prompt: str, system_prompt: str = "Você é um assistente útil.", role: str = "planner") -> str:
    """
    Faz a chamada pro OpenRouter dependendo do papel/cargo
    """
    if role == "planner": model_name = MODEL_PLANNER
    elif role == "sintetizador": model_name = MODEL_SINTETIZADOR
    elif role == "judge": model_name = MODEL_JUDGE
    else: model_name = MODEL_RESUMIDOR
    
    logger.info(f"[LLM CALL] Usando modelo: {model_name}")
        
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0 if role in ["planner", "judge"] else 0.3,
            max_tokens=2500 if role in ["sintetizador", "resumidor"] else 1000
        )
        content = response.choices[0].message.content
        return content if content is not None else "⚠️ Desculpe, o LLM falhou e retornou vazio."
    except Exception as e:
        logger.warning(f"⚠️ Falha na comunicação com OpenRouter ({model_name}): {e}. Acionando fallback secundário (OpenAI Nativa)...")
        return call_openai_fallback(prompt, system_prompt, role)

def call_openai_fallback(prompt: str, system_prompt: str, role: str) -> str:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "coloque_sua_chave_da_openai_aqui":
        logger.error("❌ Chave OPENAI_API_KEY ausente. Acionando fallback terciário (Gemini)...")
        return call_gemini_fallback(prompt, system_prompt, role)
        
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        temperature = 0.0 if role in ["planner", "judge"] else 0.3
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=2500 if role in ["sintetizador", "resumidor"] else 1000
        )
        content = response.choices[0].message.content
        return content if content is not None else "⚠️ Desculpe, o LLM falhou e retornou vazio."
    except Exception as e1:
        logger.warning(f"⚠️ Erro na execução via OpenAI ({e1}). Acionando fallback final via Gemini...")
        return call_gemini_fallback(prompt, system_prompt, role)

def call_gemini_fallback(prompt: str, system_prompt: str, role: str) -> str:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "coloque_sua_chave_do_gemini_aqui":
        logger.error("❌ GEMINI_API_KEY não configurada para o fallback!")
        return ""
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0 if "Planner" in system_prompt or "Juiz" in system_prompt else 0.3,
                max_output_tokens=2500
            )
        )
        return response.text
    except Exception as e2:
        logger.error(f"🚨 Erro Crítico: Todos os provedores de LLM falharam. Detalhe (Gemini): {e2}")
        return ""
