import os
import sys
import logging
import concurrent.futures

# Ajuste de path para import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools.parse import extract_with_docling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def worker_process_file(pdf_file, input_dir, output_dir):
    pdf_path = os.path.join(input_dir, pdf_file)
    md_filename = pdf_file.replace('.pdf', '.md')
    md_path = os.path.join(output_dir, md_filename)
    
    # Deduplicacao
    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        logger.info(f"Skipped {pdf_file}: Versao .md semantica ja existe.")
        return
        
    logger.info(f"Processando semanticamente via Docling: {pdf_file}")
    conteudo = extract_with_docling(pdf_path)
    if conteudo:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        logger.info(f"Salvo Markdown Hierarquico: {md_path}")
    else:
        logger.warning(f"Nenhum conteudo extraido para: {pdf_file}")

def executar_parser(input_dir: str, output_dir: str):
    """
    Varre os PDFs usando IBM Docling com multiprocessamento para gerar
    Markdowns ricos em semantica para RAG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    cpu_cores = max(1, (os.cpu_count() or 4) - 2)
    logger.info(f"Iniciando Docling Parser para {len(pdf_files)} arquivos usando {cpu_cores} processos paralelos.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = [executor.submit(worker_process_file, f, input_dir, output_dir) for f in pdf_files]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    PASTA_PDFS = "data/files/"
    PASTA_MD = "data/parsed/"
    executar_parser(PASTA_PDFS, PASTA_MD)
