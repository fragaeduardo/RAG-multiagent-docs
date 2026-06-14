import re
import ftfy

def restore_legal_hierarchy(text: str) -> str:
    """
    Restaura a hierarquia semântica usando regras legais (Regex) e Heurísticas Universais.
    """
    # Level 1: Documento Mãe
    # Usa o layout do Docling (deve ter #) para evitar pegar texto comum, mas força a ser H1.
    text = re.sub(r'^(#{1,6})\s+(UNIVERSIDADE FEDERAL DE GOIÁS.*|MINISTÉRIO DA EDUCAÇÃO.*|SERVIÇO PÚBLICO FEDERAL.*|REPÚBLICA FEDERATIVA DO BRASIL.*)', r'# \2', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(#{1,6})?\s*(ANEXO\s+[A-ZIVXLC0-9]+.*)', r'# \2', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(#{1,6})\s+(ESTATUTO.*|REGIMENTO.*|RESOLUÇÃO.*|EDITAL.*)', r'# \2', text, flags=re.MULTILINE | re.IGNORECASE)

    # Level 2: Título
    text = re.sub(r'^(#{1,6})\s+(TÍTULO\s+[A-ZIVXLC]+.*)', r'## \2', text, flags=re.MULTILINE)

    # Level 3: Capítulo
    text = re.sub(r'^(#{1,6})\s+(CAPÍTULO\s+[A-ZIVXLC]+.*)', r'### \2', text, flags=re.MULTILINE)

    # Level 4: Seção e Subseção
    text = re.sub(r'^(#{1,6})\s+(SEÇÃO\s+[A-ZIVXLC]+.*)', r'#### \2', text, flags=re.MULTILINE)
    text = re.sub(r'^(#{1,6})\s+(SUBSEÇÃO\s+[A-ZIVXLC]+.*)', r'#### \2', text, flags=re.MULTILINE)

    # Level 5: Artigo (Pode não ter sido pego pelo OCR, então aceitamos sem #, e forçamos para H5)
    text = re.sub(r'^(#{1,6})?\s*(Art\.\s*\S+.*)', r'##### \2', text, flags=re.MULTILINE)
    
    # Level 5: Numeração Progressiva (ex: 1.1 Objetivo).
    # Só promove se o Docling já identificou como Header (presença de #), forçando para H5.
    text = re.sub(r'^(#{1,6})\s*(\d+(?:\.\d+){0,3}\.?\s+[A-Z].*)', r'##### \2', text, flags=re.MULTILINE)

    return text

def normalize_text(text: str) -> str:
    """
    Aplica correções de codificação e restaura a estrutura hierárquica do Markdown.
    """
    clean_text = ftfy.fix_text(text)
    clean_text = restore_legal_hierarchy(clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()
