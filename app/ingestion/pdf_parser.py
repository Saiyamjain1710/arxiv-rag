from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = False
_pipeline_options.do_table_structure = False       # table model is memory-heavy; disabling as a precaution
_pipeline_options.generate_page_images = False     # don't render/keep page images in memory
_pipeline_options.generate_picture_images = False

_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)}
)

def parse_pdf_to_markdown(pdf_path: str) -> str:
    result = _converter.convert(pdf_path)
    return result.document.export_to_markdown()