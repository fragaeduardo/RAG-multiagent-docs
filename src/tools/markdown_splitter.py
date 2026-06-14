from langchain_text_splitters import MarkdownHeaderTextSplitter

def markdown_structural_split(text: str) -> list:
    """
    Divide hierarquicamente o documento com base nos Headers Markdown, mantendo a estrutura semântica nos metadados.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Artigo")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    return markdown_splitter.split_text(text)
