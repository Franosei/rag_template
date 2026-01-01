import logging
from pathlib import Path
from docx import Document as DocxDocument

from src.ingestion.extractors.common import BaseExtractor, ExtractionResult
from src.ingestion.models import Document

logger = logging.getLogger(__name__)


class DOCXExtractor(BaseExtractor):
    """Extract text from Word documents"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.docx', '.doc']
    
    async def can_extract(self, file_path: Path) -> bool:
        """Check if file is a Word document"""
        return file_path.suffix.lower() in self.supported_extensions
    
    async def extract(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str
    ) -> ExtractionResult:
        """
        Extract text from DOCX.
        
        Process:
        1. Open document
        2. Extract all paragraph text
        3. Extract table text
        4. Check for embedded images
        """
        try:
            doc = DocxDocument(file_path)
            
            # Extract paragraph text
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Extract table text
            tables = []
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    table_text.append(" | ".join(row_text))
                tables.append("\n".join(table_text))
            
            # Combine all text
            full_text = "\n\n".join(paragraphs)
            
            if tables:
                full_text += "\n\n" + "\n\n".join(tables)
            
            # Check for images
            has_images = any(
                "image" in rel.target_ref.lower()
                for rel in doc.part.rels.values()
            )
            
            has_tables = len(doc.tables) > 0
            
            # Create document metadata
            document = self._create_document_metadata(
                file_path=file_path,
                doc_id=doc_id,
                folder_id=folder_id,
                has_images=has_images,
                has_tables=has_tables
            )
            
            self.logger.info(
                f"Extracted DOCX: {file_path.name}",
                extra={
                    "paragraphs": len(paragraphs),
                    "tables": len(tables),
                    "text_length": len(full_text),
                    "has_images": has_images
                }
            )
            
            return ExtractionResult(
                document=document,
                text_content=full_text,
                success=True
            )
        
        except Exception as e:
            self.logger.error(f"DOCX extraction failed for {file_path}: {e}", exc_info=True)
            
            document = self._create_document_metadata(
                file_path=file_path,
                doc_id=doc_id,
                folder_id=folder_id
            )
            
            return ExtractionResult(
                document=document,
                text_content="",
                success=False,
                error_message=str(e)
            )