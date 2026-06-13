# O ingestor é responsável por ler os arquivos já baixados / que faltam baixar e solicitar o download usando a Skill de download


import json
import time

# Lógica modular de ingestão (conforme definido nos relatórios de estudos)
# Exemplo: _obter_lista_pdfs(), executar_download(), download_amostra()

def carregar_lista_downloads(json_path: str):
    """Lê o arquivo to_download.json"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def iniciar_ingestao():
    # TODO: Implementar lógica com tratativa de erros (blocks/403)
    pass
