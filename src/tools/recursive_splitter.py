import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = int(os.getenv("RECURSIVE_CHUNK_SIZE", "3600"))
CHUNK_OVERLAP = int(os.getenv("RECURSIVE_CHUNK_OVERLAP", "400"))

def apply_recursive_split(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """
    Estratégia de Rede de Segurança
    Garante que documentos sem formatação (ou com blocos massivos de texto) sejam 
    divididos em pedaços do tamanho ideal para o banco vetorial, clonando os 
    metadados hierárquicos para os sub-blocos resultantes.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", 
            "\n§",          # Corta em Parágrafos (símbolo §)
            "\nI - ", "\nII - ", "\nIII - ", "\nIV - ", "\nV - ", # Corta em Incisos
            "\n", 
            ". ", 
            " ", 
            ""
        ],
        length_function=len,
        is_separator_regex=False,
    )
    return recursive_splitter.split_documents(documents)
