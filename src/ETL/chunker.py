import os
import sys
import json
import logging
import concurrent.futures

# Ajuste de path para import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools.markdown_splitter import markdown_structural_split
from src.tools.recursive_splitter import apply_recursive_split
from src.tools.context_injector import apply_contextual_retrieval
from src.tools.table_unroller import unroll_markdown_tables

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_markdown_file(md_filename: str, input_dir: str, output_dir: str):
    """
    Processa um arquivo .md e o particiona em chunks semânticos.
    """
    md_path = os.path.join(input_dir, md_filename)
    document_title = md_filename.replace('.md', '')
    
    json_filename = md_filename.replace('.md', '_chunks.json')
    json_path = os.path.join(output_dir, json_filename)
    
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        logger.info(f"Skipped {md_filename}: Chunks já gerados.")
        return

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            clean_text = f.read()
            
        if not clean_text:
            return
            
        # Desdobramento Semântico de Tabelas (Anti Header-Amnesia)
        clean_text = unroll_markdown_tables(clean_text)
            
        # Divisão estrutural (Markdown Header Split)
        structural_splits = markdown_structural_split(clean_text)
        
        # Rede de Segurança (Recursive Split)
        safe_splits = apply_recursive_split(structural_splits)
        
        # Injeção de Contexto (Breadcrumbs)
        enriched_chunks = apply_contextual_retrieval(safe_splits, document_title)
        
        # Adiciona IDs unicos a cada chunk
        for i, chunk in enumerate(enriched_chunks):
            chunk['chunk_id'] = f"{document_title}_chunk_{i}"
            
        # Passo 4: Persistencia. (O Late Chunking entra na etapa de embeddings no DB)
        if enriched_chunks:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Sucesso: {len(enriched_chunks)} chunks gerados para {document_title}")
        else:
            logger.warning(f"Aviso: Nenhum chunk extraido de {document_title}")
            
    except Exception as e:
        logger.error(f"Falha ao realizar chunking de {md_filename}: {e}")

def executar_chunker(input_dir: str, output_dir: str):
    """
    Orquestrador para fatiar todos os Markdowns processados usando padrao SOTA.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    md_files = [f for f in os.listdir(input_dir) if f.endswith('.md')]
    cpu_cores = max(1, (os.cpu_count() or 4) - 2)
    logger.info(f"Iniciando SOTA Chunker para {len(md_files)} arquivos usando {cpu_cores} nucleos paralelos.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = [executor.submit(process_markdown_file, f, input_dir, output_dir) for f in md_files]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    PASTA_MD = "data/parsed/"
    PASTA_CHUNKS = "data/chunks/"
    executar_chunker(PASTA_MD, PASTA_CHUNKS)
