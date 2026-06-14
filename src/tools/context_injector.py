def apply_contextual_retrieval(splits: list, document_title: str) -> list:
    """
    Injeta os metadados hierárquicos (breadcrumbs) no conteúdo do texto do chunk.
    """
    enriched_chunks = []
    for doc in splits:
        metadata = doc.metadata
        content = doc.page_content.strip()
        
        if not content:
            continue
            
        breadcrumbs = [document_title]
        for h_level in ["Header 1", "Header 2", "Header 3", "Header 4", "Artigo"]:
            if h_level in metadata:
                breadcrumbs.append(metadata[h_level])
                
        breadcrumb_str = " > ".join(breadcrumbs)
        contextualized_content = f"[Contexto: {breadcrumb_str}]\n\n{content}"
        
        enriched_chunks.append({
            "chunk_id": None,
            "document_id": document_title,
            "breadcrumb": breadcrumb_str,
            "metadata": metadata,
            "original_content": content,
            "contextualized_content": contextualized_content,
        })
        
    return enriched_chunks
