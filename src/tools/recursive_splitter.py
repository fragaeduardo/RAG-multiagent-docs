from langchain_text_splitters import RecursiveCharacterTextSplitter

def apply_recursive_split(documents: list, chunk_size: int = 1500, chunk_overlap: int = 150) -> list:
    """
    Estratégia de Rede de Segurança
    Garante que documentos sem formatação (ou com blocos massivos de texto) sejam 
    divididos em pedaços do tamanho ideal para o banco vetorial, clonando os 
    metadados hierárquicos para os sub-blocos resultantes.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return recursive_splitter.split_documents(documents)
