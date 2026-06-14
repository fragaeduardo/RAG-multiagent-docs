import os
import sys
import json
import logging
import multiprocessing
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

# Ajuste de path para import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools.markdown_splitter import markdown_structural_split
from src.tools.recursive_splitter import apply_recursive_split
from src.tools.semantic_splitter import apply_semantic_split
from src.tools.context_injector import apply_contextual_retrieval
from src.tools.table_unroller import unroll_markdown_tables
from src.tools.chunk_cleaner import clean_and_filter_splits

from langchain_core.documents import Document

ENABLE_MARKDOWN_SPLITTER = os.getenv("ENABLE_MARKDOWN_SPLITTER", "true").lower() == "true"
ENABLE_RECURSIVE_SPLITTER = os.getenv("ENABLE_RECURSIVE_SPLITTER", "true").lower() == "true"
ENABLE_SEMANTIC_CHUNKER = os.getenv("ENABLE_SEMANTIC_CHUNKER", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_markdown_file(i: int, total: int, md_filename: str, input_dir: str, output_dir: str):
    """
    Processa um arquivo .md e o particiona em chunks semânticos e/ou estruturais.
    """
    md_path = os.path.join(input_dir, md_filename)
    document_title = md_filename.replace('.md', '')
    
    json_filename = md_filename.replace('.md', '_chunks.json')
    json_path = os.path.join(output_dir, json_filename)
    
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        logger.info(f"[{i}/{total}] Skipped {md_filename}: Chunks já gerados.")
        return

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            clean_text = f.read()
            
        if not clean_text:
            return
            
        # Desdobramento Semântico de Tabelas (Anti Header-Amnesia)
        clean_text = unroll_markdown_tables(clean_text)
            
        # Divisão estrutural (Markdown Header Split)
        if ENABLE_MARKDOWN_SPLITTER:
            structural_splits = markdown_structural_split(clean_text)
        else:
            structural_splits = [Document(page_content=clean_text, metadata={})]
        
        final_splits = []
        
        # Estrutural + Rede de Segurança (Recursive Split)
        if ENABLE_RECURSIVE_SPLITTER:
            safe_splits = apply_recursive_split(structural_splits)
            final_splits.extend(safe_splits)
        else:
            final_splits.extend(structural_splits)
            
        # Chunker Semântico paralelo (gera uma segunda via do mesmo texto para aumentar recall)
        if ENABLE_SEMANTIC_CHUNKER:
            logger.info(f"[{i}/{total}] Gerando chunks SEMÂNTICOS para {md_filename}...")
            semantic_splits = apply_semantic_split(clean_text)
            # Adiciona metadados genéricos para não quebrar o context injector
            for split in semantic_splits:
                if not split.metadata:
                    split.metadata = {}
            final_splits.extend(semantic_splits)
        
        # Filtra lixos visuais e aglutina órfãos menores que 50 chars
        final_splits = clean_and_filter_splits(final_splits, min_length=50)
        
        # Injeção de Contexto (Breadcrumbs)
        enriched_chunks = apply_contextual_retrieval(final_splits, document_title)
        
        # Adiciona IDs unicos a cada chunk
        for chunk_idx, chunk in enumerate(enriched_chunks):
            chunk['chunk_id'] = f"{document_title}_chunk_{chunk_idx}"
            
        # Passo 4: Persistencia.
        if enriched_chunks:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"[{i}/{total}] Sucesso: {len(enriched_chunks)} chunks gerados para {document_title}")
        else:
            logger.warning(f"[{i}/{total}] Aviso: Nenhum chunk extraido de {document_title}")
            
    except Exception as e:
        logger.error(f"[{i}/{total}] Falha ao realizar chunking de {md_filename}: {e}")

def executar_chunker(input_dir: str, output_dir: str):
    """
    Orquestrador para fatiar todos os Markdowns processados usando padrao SOTA.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    md_files = [f for f in os.listdir(input_dir) if f.endswith('.md')]
    
    if ENABLE_SEMANTIC_CHUNKER:
        cpu_cores = int(os.getenv("MAX_PARALLEL_WORKERS", "3"))
        logger.info(f"Semantic Chunker ATIVADO! Acelerando via ProcessPool com {cpu_cores} workers paralelos. {len(md_files)} arquivos pendentes.")
        ctx = multiprocessing.get_context('spawn')
    else:
        cpu_cores = max(1, (os.cpu_count() or 4) - 2)
        logger.info(f"Iniciando Chunker Estrutural para {len(md_files)} arquivos usando {cpu_cores} nucleos paralelos.")
        ctx = multiprocessing.get_context('fork')
    
    total_files = len(md_files)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores, mp_context=ctx) as executor:
        futures = [executor.submit(process_markdown_file, i, total_files, f, input_dir, output_dir) for i, f in enumerate(md_files, 1)]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    PASTA_MD = "data/parsed/"
    PASTA_CHUNKS = "data/chunks/"
    executar_chunker(PASTA_MD, PASTA_CHUNKS)
