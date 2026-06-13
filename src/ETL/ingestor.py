# Orquestrador de ETL para ingestão de documentos

import json
import os
import logging
import sys

# Ajustar path para importar a tool a partir do root do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools.download import download_pdf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def executar_ingestao(json_path: str, output_dir: str):
    """Lê o arquivo JSON de manifestos e executa os downloads pendentes."""
    if not os.path.exists(json_path):
        logging.error(f"Arquivo {json_path} não encontrado.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    documentos = manifest.get('documents', [])
    pendentes = [doc for doc in documentos if doc.get('status') == 'pending']
    
    logging.info(f"Iniciando ingestão. Total: {len(documentos)} | Pendentes: {len(pendentes)}")
    
    sucessos = 0
    falhas = 0
    
    total_pendentes = len(pendentes)
    for idx, doc in enumerate(pendentes, start=1):
        doc_id = doc.get('id', 'doc_desconhecido')
        url = doc.get('url', '')
        if not url:
            continue
            
        filename = f"{doc_id}.pdf"
        file_path = os.path.join(output_dir, filename)
        
        # Controle de deduplicacao via checagem em disco
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
            logging.info(f"[{idx}/{total_pendentes}] Skipped {doc_id}: Arquivo existente")
            doc['status'] = 'downloaded'
            sucessos += 1
            continue
            
        logging.info(f"[{idx}/{total_pendentes}] Iniciando download: {doc_id}")
        sucesso = download_pdf(url, file_path)
        
        if sucesso:
            doc['status'] = 'downloaded'
            sucessos += 1
        else:
            doc['status'] = 'failed'
            falhas += 1
            
        # Atualiza o arquivo de estado para garantir persistencia
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            
    logging.info(f"Processo finalizado. Sucessos: {sucessos} | Falhas: {falhas}")

if __name__ == "__main__":
    MANIFESTO = "data/to_download.json"
    PASTA_DESTINO = "data/files/"
    executar_ingestao(MANIFESTO, PASTA_DESTINO)
