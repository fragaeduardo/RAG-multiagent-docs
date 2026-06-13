import logging
import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_with_docling(pdf_path: str) -> str:
    """
    Usa o IBM Docling para converter PDFs mantendo 100% da hierarquia 
    semantica (H1, H2, H3), tabelas complexas, layout e acionando OCR automaticamente.
    """
    if not os.path.exists(pdf_path):
        return ""
        
    try:
        # Configurando Docling para MAXIMA QUALIDADE (Forcando OCR em imagens e layout complexo)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_page_images = False

        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Falha critica no Docling ({pdf_path}): {e}")
        return ""