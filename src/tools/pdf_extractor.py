import logging
import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    AcceleratorOptions, 
    AcceleratorDevice,
    TableFormerMode
)
from docling.datamodel.base_models import InputFormat

logger = logging.getLogger(__name__)

_GLOBAL_CONVERTER = None

def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def get_converter() -> DocumentConverter:
    global _GLOBAL_CONVERTER
    if _GLOBAL_CONVERTER is None:
        device = AcceleratorDevice.CUDA if _cuda_available() else AcceleratorDevice.CPU
        logger.info(f"Iniciando Docling Converter com hardware: {device.name}")
        
        accel_options = AcceleratorOptions(
            num_threads=os.cpu_count() or 4,
            device=device
        )
        
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accel_options,
            do_ocr=False,
            do_table_structure=True,
            generate_page_images=False
        )
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
        _GLOBAL_CONVERTER = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    return _GLOBAL_CONVERTER

def extract_with_docling(pdf_path: str) -> str:
    """
    Converte o arquivo PDF mantendo hierarquia semântica, tabelas complexas e layout.
    """
    if not os.path.exists(pdf_path):
        return ""
        
    try:
        converter = get_converter()
        result = converter.convert(pdf_path, raises_on_error=False)
        return result.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Falha na extração de conteúdo ({pdf_path}): {e}")
        return ""
    finally:
        import gc
        gc.collect()
        if _cuda_available():
            import torch
            torch.cuda.empty_cache()
