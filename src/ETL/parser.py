import os
import sys
import logging

# ── Configuração de Ambiente ────────────────────────────────────────────────

# Adiciona o diretório base ao path do sistema para importações internas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools.pdf_extractor import extract_with_docling
from src.tools.text_normalizer import normalize_text

# ── Configuração de Logs ────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Silencia avisos de depreciação e logs de bibliotecas externas
logging.getLogger("docling").setLevel(logging.ERROR)
logging.getLogger("docling.pipeline.standard_pdf_pipeline").setLevel(logging.ERROR)
logging.getLogger("RapidOCR").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ── Fluxo Principal de Processamento ────────────────────────────────────────

import sys
import multiprocessing
import concurrent.futures

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def _processar_unico_arquivo(args):
    i, total, pdf_file, input_dir, output_dir = args
    pdf_path = os.path.join(input_dir, pdf_file)
    md_filename = pdf_file.replace('.pdf', '.md')
    md_path = os.path.join(output_dir, md_filename)
    
    logger.info(f"[{i}/{total}] INICIANDO: Extração de {pdf_file}")
    raw_markdown = extract_with_docling(pdf_path)
    conteudo = normalize_text(raw_markdown) if raw_markdown else ""
    
    if conteudo:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        logger.info(f"[{i}/{total}] SUCESSO: Estrutura Markdown salva para {pdf_file}")
    else:
        logger.warning(f"[{i}/{total}] FALHA: Conteúdo não extraído para {pdf_file}")

def executar_parser(input_dir: str, output_dir: str):
    """
    Coordena a extração semântica em lote dos documentos originais via processamento paralelo.
    """
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Memória da GPU limpa antes de iniciar o processamento.")
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    
    # Filtra arquivos pendentes
    pdf_files = []
    for f in os.listdir(input_dir):
        if f.endswith('.pdf'):
            md_path = os.path.join(output_dir, f.replace('.pdf', '.md'))
            if not (os.path.exists(md_path) and os.path.getsize(md_path) > 0):
                pdf_files.append(f)
    
    total = len(pdf_files)
    if total == 0:
        logger.info("Todos os documentos já foram processados! Nada a fazer.")
        return

    WORKERS_SEGUROS = int(os.getenv("MAX_PARALLEL_WORKERS", 3))
    logger.info(f"Iniciando processamento para {total} documento(s) PENDENTES com {WORKERS_SEGUROS} processos. (Limite seguro para VRAM)")
    
    tarefas = [(i, total, f, input_dir, output_dir) for i, f in enumerate(pdf_files, 1)]
    
    # IMPORTANTE: Para evitar que os processos fiquem presos na VRAM como zumbis (CUDA OOM),
    # é OBRIGATÓRIO usar o contexto 'spawn' no Linux ao invés do 'fork' padrão.
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS_SEGUROS, mp_context=ctx) as executor:
        executor.map(_processar_unico_arquivo, tarefas)

# ── Execução ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PASTA_PDFS = "data/files/"
    PASTA_MD = "data/parsed/"
    executar_parser(PASTA_PDFS, PASTA_MD)
