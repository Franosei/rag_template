import logging
from pathlib import Path
import openpyxl
from typing import List, Dict, Any

from src.ingestion.extractors.common import BaseExtractor, ExtractionResult
from src.ingestion.models import Document, Chunk, Modality, ExtractionMethod, ConfidenceLevel, TableMetadata
from src.utils.ids import generate_chunk_id
import hashlib

logger = logging.getLogger(__name__)


class XLSXExtractor(BaseExtractor):
    """
    Extract data from Excel files.
    Treats each sheet as a table and creates table chunks.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.xlsx', '.xls', '.csv']
    
    async def can_extract(self, file_path: Path) -> bool:
        """Check if file is an Excel file"""
        return file_path.suffix.lower() in self.supported_extensions
    
    async def extract(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str
    ) -> ExtractionResult:
        """
        Extract tables from Excel file.
        
        Each sheet becomes:
        1. A table chunk with structured data
        2. Linearized text representation for vector search
        """
        try:
            if file_path.suffix.lower() == '.csv':
                return await self._extract_csv(file_path, doc_id, folder_id)
            else:
                return await self._extract_xlsx(file_path, doc_id, folder_id)
        
        except Exception as e:
            self.logger.error(f"Excel extraction failed for {file_path}: {e}", exc_info=True)
            
            document = self._create_document_metadata(
                file_path=file_path,
                doc_id=doc_id,
                folder_id=folder_id,
                has_tables=True
            )
            
            return ExtractionResult(
                document=document,
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def _extract_xlsx(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str
    ) -> ExtractionResult:
        """Extract from XLSX/XLS file"""
        
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        
        chunks = []
        text_parts = []
        table_index = 0
        
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            
            # Extract sheet data
            sheet_data = []
            headers = []
            
            for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
                # Filter out empty rows
                row_data = [str(cell) if cell is not None else "" for cell in row]
                
                if not any(cell.strip() for cell in row_data):
                    continue
                
                if row_idx == 1:
                    # First row as headers
                    headers = row_data
                else:
                    sheet_data.append(row_data)
            
            if not sheet_data:
                continue  # Skip empty sheets
            
            # Convert to structured format
            structured_data = []
            for row in sheet_data:
                row_dict = {}
                for i, value in enumerate(row):
                    header = headers[i] if i < len(headers) else f"Column_{i+1}"
                    row_dict[header] = value
                structured_data.append(row_dict)
            
            # Create linearized text representation
            table_text = f"Sheet: {sheet_name}\n"
            table_text += " | ".join(headers) + "\n"
            table_text += "-" * 80 + "\n"
            
            for row in sheet_data[:50]:  # First 50 rows
                table_text += " | ".join(row) + "\n"
            
            text_parts.append(table_text)
            
            # Create table chunk
            chunk_id = generate_chunk_id(doc_id, table_index)
            content_hash = hashlib.md5(table_text.encode()).hexdigest()
            
            chunk = Chunk(
                chunk_id=chunk_id,
                folder_id=folder_id,
                source_doc_id=doc_id,
                modality=Modality.TABLE,
                content_text=table_text,
                file_path=str(file_path),
                file_name=file_path.name,
                chunk_index=table_index,
                extraction_method=ExtractionMethod.TABLE_PARSE,
                confidence=ConfidenceLevel.HIGH,
                table_metadata=TableMetadata(
                    rows=len(sheet_data),
                    columns=len(headers),
                    headers=headers,
                    table_index=table_index,
                    structured_data=structured_data[:100]  # Limit to 100 rows
                ),
                metadata={"sheet_name": sheet_name},
                content_hash=content_hash
            )
            
            chunks.append(chunk)
            table_index += 1
        
        wb.close()
        
        # Create document metadata
        document = self._create_document_metadata(
            file_path=file_path,
            doc_id=doc_id,
            folder_id=folder_id,
            has_tables=True,
            table_chunks=len(chunks)
        )
        
        full_text = "\n\n".join(text_parts)
        
        self.logger.info(
            f"Extracted Excel: {file_path.name}",
            extra={
                "sheets": len(wb.sheetnames),
                "tables": len(chunks),
                "text_length": len(full_text)
            }
        )
        
        return ExtractionResult(
            document=document,
            text_content=full_text,
            chunks=chunks,
            success=True
        )
    
    async def _extract_csv(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str
    ) -> ExtractionResult:
        """Extract from CSV file"""
        import csv
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            document = self._create_document_metadata(
                file_path=file_path,
                doc_id=doc_id,
                folder_id=folder_id,
                has_tables=True
            )
            
            return ExtractionResult(
                document=document,
                text_content="",
                success=True
            )
        
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
        
        # Create structured data
        structured_data = []
        for row in data_rows:
            row_dict = {}
            for i, value in enumerate(row):
                header = headers[i] if i < len(headers) else f"Column_{i+1}"
                row_dict[header] = value
            structured_data.append(row_dict)
        
        # Create linearized text
        table_text = " | ".join(headers) + "\n"
        table_text += "-" * 80 + "\n"
        
        for row in data_rows[:50]:  # First 50 rows
            table_text += " | ".join(row) + "\n"
        
        # Create table chunk
        chunk_id = generate_chunk_id(doc_id, 0)
        content_hash = hashlib.md5(table_text.encode()).hexdigest()
        
        chunk = Chunk(
            chunk_id=chunk_id,
            folder_id=folder_id,
            source_doc_id=doc_id,
            modality=Modality.TABLE,
            content_text=table_text,
            file_path=str(file_path),
            file_name=file_path.name,
            chunk_index=0,
            extraction_method=ExtractionMethod.TABLE_PARSE,
            confidence=ConfidenceLevel.HIGH,
            table_metadata=TableMetadata(
                rows=len(data_rows),
                columns=len(headers),
                headers=headers,
                table_index=0,
                structured_data=structured_data[:100]
            ),
            content_hash=content_hash
        )
        
        document = self._create_document_metadata(
            file_path=file_path,
            doc_id=doc_id,
            folder_id=folder_id,
            has_tables=True,
            table_chunks=1
        )
        
        self.logger.info(
            f"Extracted CSV: {file_path.name}",
            extra={
                "rows": len(data_rows),
                "columns": len(headers)
            }
        )
        
        return ExtractionResult(
            document=document,
            text_content=table_text,
            chunks=[chunk],
            success=True
        )