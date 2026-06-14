import re
from langchain_core.documents import Document

def clean_and_filter_splits(splits: list[Document], min_length: int = 40) -> list[Document]:
    """
    Higieniza a pool de chunks gerada pelos cortadores.
    - Remove chunks inúteis (ex: placeholders `<!-- image -->`, separadores isolados).
    - Aglutina chunks órfãos (muito curtos) com o chunk anterior, respeitando os metadados.
    """
    cleaned = []
    
    for split in splits:
        text = split.page_content.strip()
        
        # Remove lixos de formatação e placeholders puros
        text = re.sub(r'<!--\s*image\s*-->', '', text, flags=re.IGNORECASE).strip()
        
        # Remove linhas que são apenas traços, asteriscos ou pipes isolados
        text = re.sub(r'^[\s\*\-\|]+$', '', text, flags=re.MULTILINE).strip()
        
        # Comprime espaços em branco duplos (quebras de tabulação de PDFs) mas mantém as quebras de linha (\n)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        
        if not text:
            continue
            
        # Filtro de Entropia: Se o chunk NÃO tiver nenhuma letra do alfabeto (for só números, datas ou códigos)
        # Ele não possui valor semântico isolado no RAG. O unroller de tabelas já injeta letras (ex: "Valor: 100").
        if not re.search(r'[a-zA-Z]', text):
            continue
            
        # Filtro de Rodapé (Amnésia de Paginação): Docling às vezes falha em omitir paginação
        if re.match(r'^(Página|Page)\s*\d+(\s*(de|/)\s*\d+)?$', text, flags=re.IGNORECASE):
            continue
            
        # Aglutinação de Órfãos: Se o chunk for curtinho (ex: um título perdido, uma frase solta)
        if len(text) < min_length and cleaned:
            prev = cleaned[-1]
            # Só aglutina se vierem do mesmo nível de metadados
            if prev.metadata == split.metadata:
                prev.page_content += "\n\n" + text
                continue
                
        # Atualiza e salva se for válido
        split.page_content = text
        cleaned.append(split)
        
    return cleaned
