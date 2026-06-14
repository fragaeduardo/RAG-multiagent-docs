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

def merge_consecutive_headers(text: str) -> str:
    """
    Junta cabeçalhos consecutivos do mesmo nível (ex: múltiplos '#' no topo do documento)
    em uma única linha separada por hífen. Isso enriquece os metadados do chunker.
    """
    lines = text.split('\n')
    merged_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged_lines.append(line)
            continue
            
        match = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if match:
            level = match.group(1)
            content = match.group(2)
            
            # Acha a última linha não vazia
            last_non_empty_idx = -1
            for i in range(len(merged_lines)-1, -1, -1):
                if merged_lines[i].strip():
                    last_non_empty_idx = i
                    break
                    
            if last_non_empty_idx != -1:
                last_line = merged_lines[last_non_empty_idx].strip()
                last_match = re.match(r'^(#{1,6})\s+(.*)', last_line)
                
                # Se for o mesmo nível do cabeçalho imediatamente anterior (ignorando espaços)
                if last_match and last_match.group(1) == level:
                    merged_content = last_match.group(2) + " - " + content
                    # Atualiza a linha anterior com o texto combinado
                    merged_lines[last_non_empty_idx] = f"{level} {merged_content}"
                    continue # Pula a inserção de uma nova linha
                    
        merged_lines.append(line)
        
    return '\n'.join(merged_lines)

def normalize_text(text: str) -> str:
    """
    Aplica correções de codificação e restaura a estrutura hierárquica do Markdown.
    """
    clean_text = ftfy.fix_text(text)
    clean_text = restore_legal_hierarchy(clean_text)
    clean_text = merge_consecutive_headers(clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()
