import re

# Quebra a linha da tabela pelo pipe
def _parse_row(row_str):
    # Remove leading and trailing pipes
    row_str = row_str.strip()
    if row_str.startswith('|'):
        row_str = row_str[1:]
    if row_str.endswith('|'):
        row_str = row_str[:-1]
    
    # Split by pipe and strip, convertendo <br> de células multilinha em espaços
    return [re.sub(r'<br\s*/?>', ' ', cell, flags=re.IGNORECASE).strip() for cell in row_str.split('|')]

def unroll_markdown_tables(text: str) -> str:
    """
    Transforma as tabelas do markdown em texto corrido (Cabeçalho: Valor)
    pra facilitar a vida do modelo de embedding.
    """
    lines = text.split('\n')
    unrolled_lines = []
    
    in_table = False
    table_headers = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Checa se achou uma tabela nova (linha com pipe e depois o separador |---|)
        if not in_table and stripped.startswith('|') and i + 1 < len(lines) and re.match(r'^\|?[\s\-:]+\|[\s\-:\|]+$', lines[i+1].strip()):
            in_table = True
            table_headers = _parse_row(line)
            i += 1 # Pula a linha atual (cabeçalho)
            i += 1 # Pula a linha do separador (|---|)
            unrolled_lines.append("\n[Início de Dados Tabulares]")
            continue
            
        if in_table:
            # Continua lendo a tabela
            if stripped.startswith('|'):
                # É uma linha de dados
                row_cells = _parse_row(line)
                
                # Monta a frase da linha
                frases = []
                for col_idx, cell_value in enumerate(row_cells):
                    if not cell_value or cell_value == '-':
                        continue # Pula células vazias
                    
                    # Pega o cabeçalho correspondente (se existir)
                    header = table_headers[col_idx] if col_idx < len(table_headers) else f"Coluna {col_idx+1}"
                    
                    if not header:
                        frases.append(f"{cell_value}")
                    else:
                        frases.append(f"{header}: {cell_value}")
                
                if frases:
                    linha_desdobrada = " * " + ". ".join(frases) + "."
                    # Ignora a linha se for apenas traços (resíduo de formatador markdown quebrado)
                    if re.match(r'^[\s\*\-\.:]+$', linha_desdobrada):
                        continue
                    # Remove pontos duplos ou finais mal formatados
                    linha_desdobrada = re.sub(r'\.\.', '.', linha_desdobrada)
                    unrolled_lines.append(linha_desdobrada)
            else:
                # Tabela acabou (encontrou uma linha sem pipe)
                # Nota: linhas em branco podem ou não significar o fim da tabela dependendo do parser, mas tabelas normais não têm quebra.
                if stripped == '':
                    # Dá um desconto pra linhas em branco no meio da tabela (Docling vacila às vezes)
                    # Dá uma espiada na próxima linha pra ver se a tabela continua
                    next_is_table = False
                    for j in range(i+1, min(i+3, len(lines))):
                        if lines[j].strip().startswith('|'):
                            next_is_table = True
                            break
                    if next_is_table:
                        i += 1
                        continue
                
                in_table = False
                unrolled_lines.append("[Fim dos Dados Tabulares]\n")
                unrolled_lines.append(line) # Adiciona a linha atual que quebrou a tabela
        else:
            unrolled_lines.append(line)
            
        i += 1
        
    # Caso o arquivo acabe com a tabela ainda aberta
    if in_table:
        unrolled_lines.append("[Fim dos Dados Tabulares]\n")
        
    return '\n'.join(unrolled_lines)
