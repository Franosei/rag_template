# src/ingestion/pipeline.py

import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.ingestion.extractors.pdf import PDFExtractor
from src.ingestion.extractors.docx import DOCXExtractor
from src.ingestion.extractors.xlsx import XLSXExtractor
from src.ingestion.extractors.common import BaseExtractor, ExtractionResult

from src.ingestion.extractors.images import (
    ImageExtractor,
    PDFImageExtractor,
    DOCXImageExtractor,
)

from src.ingestion.chunking.chunker import TextChunker
from src.ingestion.models import Chunk, Document
from src.core.policies.folder_policy import FolderPolicy
from src.utils.ids import generate_document_id
from src.utils.fileio import save_json
from src.app.settings import settings
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Mission:
    - Convert heterogeneous folder content (PDF, DOCX, XLSX/CSV, images, text files)
      into a unified set of retrievable, traceable chunks.

    Vision:
    - Evidence-first processing with strong provenance:
      every chunk must point back to an originating document location.

    Process:
    1) Extract text/tables from documents
    2) Extract and process images (OCR + caption) guided by FolderPolicy
    3) Chunk text into retrievable units
    4) Persist processed chunks + document metadata to disk (local dev)
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".csv",
        ".txt",
        ".md",
        ".rst",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
        ".gif",
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        data_dir: Optional[Path] = None,
    ):
        self.llm_client = llm_client
        self.data_dir = data_dir or settings.data_dir

        # Text/table extractors
        self.text_extractors: List[BaseExtractor] = [
            PDFExtractor(),
            DOCXExtractor(),
            XLSXExtractor(),
        ]

        # Image processing (OCR + caption), guided by FolderPolicy at runtime
        self.image_extractor = ImageExtractor(
            enable_ocr=True,
            enable_caption=True,
            llm_client=llm_client,
        )
        self.pdf_image_extractor = PDFImageExtractor(self.image_extractor)
        self.docx_image_extractor = DOCXImageExtractor(self.image_extractor)

        # Chunker
        self.chunker = TextChunker()

        # Output directory (local dev)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    async def ingest_folder(self, folder_path: Path, policy: FolderPolicy) -> Dict[str, Any]:
        """
        Ingest an entire folder according to its FolderPolicy.

        Returns ingestion statistics and output locations.
        """
        start_time = datetime.utcnow()

        logger.info(
            "Starting folder ingestion",
            extra={"folder_path": str(folder_path), "folder_id": policy.folder_id},
        )

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        files_to_process = self._collect_files(folder_path)

        logger.info(
            "Collected files to process",
            extra={"folder": folder_path.name, "total_files": len(files_to_process)},
        )

        all_documents: List[Document] = []
        all_chunks: List[Chunk] = []

        # NOTE: kept sequential for determinism; can be parallelized later with bounded concurrency
        for file_path in files_to_process:
            result = await self.ingest_document(file_path=file_path, policy=policy)

            if result.get("success"):
                doc = result.get("document")
                chunks = result.get("chunks") or []
                if doc is not None:
                    all_documents.append(doc)
                all_chunks.extend(chunks)
            else:
                logger.warning(
                    "Document ingestion failed",
                    extra={"file": str(file_path), "error": result.get("error")},
                )

        # Persist results
        output_path = self.processed_dir / policy.folder_id
        output_path.mkdir(parents=True, exist_ok=True)

        chunks_file = output_path / "chunks.json"
        docs_file = output_path / "documents.json"

        save_json([c.model_dump(mode="json") for c in all_chunks], chunks_file)
        save_json([d.model_dump(mode="json") for d in all_documents], docs_file)

        duration = (datetime.utcnow() - start_time).total_seconds()

        stats = {
            "folder_id": policy.folder_id,
            "folder_path": str(folder_path),
            "total_files": len(files_to_process),
            "processed_documents": len(all_documents),
            "total_chunks": len(all_chunks),
            "text_chunks": sum(1 for c in all_chunks if c.modality == "text"),
            "table_chunks": sum(1 for c in all_chunks if c.modality == "table"),
            "image_chunks": sum(1 for c in all_chunks if c.modality == "image"),
            "duration_seconds": duration,
            "output_path": str(output_path),
            "chunks_file": str(chunks_file),
            "documents_file": str(docs_file),
        }

        logger.info("Folder ingestion complete", extra=stats)
        return stats

    async def ingest_document(self, file_path: Path, policy: FolderPolicy) -> Dict[str, Any]:
        """
        Ingest a single file.

        Steps:
        1) Extract text/tables
        2) Chunk text
        3) Extract images (if policy requires)
        4) Combine chunks and update Document metadata
        """
        logger.info("Ingesting document", extra={"file": file_path.name})

        doc_id = generate_document_id(file_path)
        folder_id = policy.folder_id

        all_chunks: List[Chunk] = []
        document: Optional[Document] = None

        try:
            extraction_result = await self._extract_content(file_path, doc_id, folder_id)

            document = extraction_result.document

            if not extraction_result.success:
                return {
                    "success": False,
                    "document": document,
                    "chunks": [],
                    "error": extraction_result.error_message or "Extraction failed",
                }

            # Pre-chunked content (e.g., Excel tables) from extractor
            all_chunks.extend(extraction_result.chunks or [])

            # Chunk extracted text
            if extraction_result.text_content and extraction_result.text_content.strip():
                text_chunks = self.chunker.chunk_text(
                    text=extraction_result.text_content,
                    doc_id=doc_id,
                    folder_id=folder_id,
                    file_path=str(file_path),
                    file_name=file_path.name,
                )
                all_chunks.extend(text_chunks)

            # Extract/process images if required by folder policy
            if policy.image_processing_required and document is not None:
                image_chunks = await self._extract_images(
                    file_path=file_path,
                    doc_id=doc_id,
                    folder_id=folder_id,
                    policy=policy,
                    document=document,
                )
                all_chunks.extend(image_chunks)

            # Update document stats
            if document is not None:
                document.total_chunks = len(all_chunks)
                document.text_chunks = sum(1 for c in all_chunks if c.modality == "text")
                document.table_chunks = sum(1 for c in all_chunks if c.modality == "table")
                document.image_chunks = sum(1 for c in all_chunks if c.modality == "image")
                document.processed_at = datetime.utcnow()

            logger.info(
                "Document ingested",
                extra={
                    "file": file_path.name,
                    "total_chunks": len(all_chunks),
                    "text_chunks": sum(1 for c in all_chunks if c.modality == "text"),
                    "table_chunks": sum(1 for c in all_chunks if c.modality == "table"),
                    "image_chunks": sum(1 for c in all_chunks if c.modality == "image"),
                },
            )

            return {"success": True, "document": document, "chunks": all_chunks}

        except Exception as e:
            logger.error("Document ingestion failed", extra={"file": str(file_path)}, exc_info=True)
            return {"success": False, "document": document, "chunks": all_chunks, "error": str(e)}

    async def _extract_content(self, file_path: Path, doc_id: str, folder_id: str) -> ExtractionResult:
        """
        Extract text/tables from a file using the first compatible extractor.

        Falls back to plain text for .txt/.md/.rst.
        Returns a failed ExtractionResult for unsupported types.
        """
        for extractor in self.text_extractors:
            try:
                if await extractor.can_extract(file_path):
                    return await extractor.extract(file_path, doc_id, folder_id)
            except Exception as e:
                logger.warning(
                    "Extractor failed; trying next",
                    extra={"extractor": extractor.__class__.__name__, "file": str(file_path), "error": str(e)},
                )

        # Plain text fallback
        if file_path.suffix.lower() in {".txt", ".md", ".rst"}:
            base = BaseExtractor()
            document = base._create_document_metadata(file_path, doc_id, folder_id)
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                return ExtractionResult(document=document, text_content=text, chunks=[], success=True)
            except Exception as e:
                logger.error("Plain text extraction failed", extra={"file": str(file_path), "error": str(e)})
                return ExtractionResult(
                    document=document,
                    text_content="",
                    chunks=[],
                    success=False,
                    error_message=f"Plain text extraction failed: {e}",
                )

        # Unsupported file type
        base = BaseExtractor()
        document = base._create_document_metadata(file_path, doc_id, folder_id)
        return ExtractionResult(
            document=document,
            text_content="",
            chunks=[],
            success=False,
            error_message="Unsupported file type",
        )

    async def _extract_images(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str,
        policy: FolderPolicy,
        document: Document,
    ) -> List[Chunk]:
        """
        Extract images according to policy settings.

        - Updates ImageExtractor toggles from FolderPolicy:
          OCR/caption enablement and confidence threshold.
        - Routes to appropriate image extractor based on file type.
        """
        # Update image extractor settings from policy (trust policy)
        # NOTE: these attributes must exist in ImageExtractor implementation
        self.image_extractor.enable_ocr = bool(policy.ocr_enabled)
        self.image_extractor.enable_caption = bool(policy.caption_enabled)
        self.image_extractor.min_confidence = float(policy.min_confidence_threshold)

        ext = file_path.suffix.lower()

        try:
            if ext == ".pdf":
                chunks = await self.pdf_image_extractor.extract_images_from_pdf(
                    pdf_path=file_path,
                    doc_id=doc_id,
                    folder_id=folder_id,
                    is_scanned=getattr(document, "is_scanned", False),
                )
                logger.info("Extracted image chunks from PDF", extra={"file": file_path.name, "count": len(chunks)})
                return chunks

            if ext in {".docx", ".doc"}:
                chunks = await self.docx_image_extractor.extract_images_from_docx(
                    docx_path=file_path,
                    doc_id=doc_id,
                    folder_id=folder_id,
                )
                logger.info("Extracted image chunks from DOCX", extra={"file": file_path.name, "count": len(chunks)})
                return chunks

            if ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}:
                chunks = await self.image_extractor.extract_from_file(
                    image_path=file_path,
                    doc_id=doc_id,
                    folder_id=folder_id,
                )
                logger.info("Extracted standalone image chunks", extra={"file": file_path.name, "count": len(chunks)})
                return chunks

            return []

        except Exception as e:
            logger.error("Image extraction failed", extra={"file": file_path.name, "error": str(e)}, exc_info=True)
            return []

    def _collect_files(self, folder_path: Path) -> List[Path]:
        """
        Collect all supported files from the folder recursively.

        Hidden files and folders (starting with '.') are skipped.
        """
        files: List[Path] = []

        for item in folder_path.rglob("*"):
            if not item.is_file():
                continue

            # Skip hidden files/dirs
            if item.name.startswith(".") or any(part.startswith(".") for part in item.parts):
                continue

            if item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(item)

        return sorted(files)
