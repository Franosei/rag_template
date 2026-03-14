"""End-to-end folder ingestion pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.app.settings import settings
from src.core.policies.folder_policy import FolderPolicy
from src.ingestion.chunking.chunker import TextChunker
from src.ingestion.extractors.common import BaseExtractor, ExtractionResult
from src.ingestion.extractors.docx import DOCXExtractor
from src.ingestion.extractors.images import DOCXImageExtractor, ImageExtractor, PDFImageExtractor
from src.ingestion.extractors.pdf import PDFExtractor
from src.ingestion.extractors.xlsx import XLSXExtractor
from src.ingestion.models import Chunk, Document
from src.llm.client import LLMClient
from src.utils.fileio import save_json
from src.utils.ids import generate_document_id

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Convert supported files into processed documents and retrievable chunks."""

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".xlsx",
        ".xls",
        ".csv",
        ".txt",
        ".md",
        ".rst",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    }

    def __init__(self, llm_client: LLMClient | None = None, data_dir: Path | None = None):
        self.llm_client = llm_client
        self.data_dir = data_dir or settings.data_dir
        self.text_extractors: list[BaseExtractor] = [
            PDFExtractor(),
            DOCXExtractor(),
            XLSXExtractor(),
        ]
        self.image_extractor = ImageExtractor(enable_ocr=False, enable_caption=False, llm_client=llm_client)
        self.pdf_image_extractor = PDFImageExtractor(self.image_extractor)
        self.docx_image_extractor = DOCXImageExtractor(self.image_extractor)
        self.chunker = TextChunker()

    async def ingest_folder(self, folder_path: Path, policy: FolderPolicy) -> dict[str, object]:
        """Ingest every supported file in a folder and persist processed outputs."""

        started_at = datetime.now(tz=timezone.utc)
        files_to_process = self._collect_files(folder_path)
        all_documents: list[Document] = []
        all_chunks: list[Chunk] = []

        for file_path in files_to_process:
            result = await self.ingest_document(file_path=file_path, policy=policy)
            if result["success"]:
                document = result["document"]
                if document is not None:
                    all_documents.append(document)
                all_chunks.extend(result["chunks"])
            else:
                logger.warning(
                    "Document ingestion failed",
                    extra={"file": str(file_path), "error": result.get("error")},
                )

        output_path = self.data_dir / "processed" / policy.folder_id
        output_path.mkdir(parents=True, exist_ok=True)
        chunks_file = output_path / "chunks.json"
        documents_file = output_path / "documents.json"
        save_json([chunk.model_dump(mode="json") for chunk in all_chunks], chunks_file)
        save_json([document.model_dump(mode="json") for document in all_documents], documents_file)

        duration_seconds = round((datetime.now(tz=timezone.utc) - started_at).total_seconds(), 3)
        return {
            "folder_id": policy.folder_id,
            "folder_path": str(folder_path.resolve()),
            "total_files": len(files_to_process),
            "processed_documents": len(all_documents),
            "total_chunks": len(all_chunks),
            "text_chunks": sum(1 for chunk in all_chunks if chunk.modality.value == "text"),
            "table_chunks": sum(1 for chunk in all_chunks if chunk.modality.value == "table"),
            "image_chunks": sum(1 for chunk in all_chunks if chunk.modality.value == "image"),
            "duration_seconds": duration_seconds,
            "output_path": str(output_path),
            "chunks_file": str(chunks_file),
            "documents_file": str(documents_file),
        }

    async def ingest_document(self, file_path: Path, policy: FolderPolicy) -> dict[str, object]:
        """Ingest one file and return its document record plus created chunks."""

        doc_id = generate_document_id(file_path)
        folder_id = policy.folder_id

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

            all_chunks = list(extraction_result.chunks)
            if extraction_result.text_content.strip():
                all_chunks.extend(
                    self.chunker.chunk_text(
                        text=extraction_result.text_content,
                        doc_id=doc_id,
                        folder_id=folder_id,
                        file_path=str(file_path.resolve()),
                        file_name=file_path.name,
                    )
                )

            if policy.image_processing_required:
                image_chunks = await self._extract_images(file_path, doc_id, folder_id, policy, document)
                all_chunks.extend(image_chunks)

            document.total_chunks = len(all_chunks)
            document.text_chunks = sum(1 for chunk in all_chunks if chunk.modality.value == "text")
            document.table_chunks = sum(1 for chunk in all_chunks if chunk.modality.value == "table")
            document.image_chunks = sum(1 for chunk in all_chunks if chunk.modality.value == "image")
            document.processed_at = datetime.now(tz=timezone.utc)

            return {"success": True, "document": document, "chunks": all_chunks}
        except Exception as exc:
            logger.exception("Document ingestion failed", extra={"file": str(file_path)})
            return {"success": False, "document": None, "chunks": [], "error": str(exc)}

    async def _extract_content(self, file_path: Path, doc_id: str, folder_id: str) -> ExtractionResult:
        """Extract content with the first compatible extractor, then plain-text fallback."""

        for extractor in self.text_extractors:
            if await extractor.can_extract(file_path):
                result = await extractor.extract(file_path, doc_id, folder_id)
                if result.success:
                    return result
                logger.info(
                    "Extractor returned unsuccessful result",
                    extra={"extractor": extractor.__class__.__name__, "file": str(file_path), "error": result.error_message},
                )
                return result

        if file_path.suffix.lower() in {".txt", ".md", ".rst"}:
            document = self.text_extractors[0]._create_document_metadata(file_path, doc_id, folder_id)
            try:
                return ExtractionResult(
                    document=document,
                    text_content=file_path.read_text(encoding="utf-8", errors="ignore"),
                    success=True,
                )
            except Exception as exc:
                return ExtractionResult(document=document, text_content="", success=False, error_message=str(exc))

        document = self.text_extractors[0]._create_document_metadata(file_path, doc_id, folder_id)
        return ExtractionResult(document=document, text_content="", success=False, error_message="Unsupported file type")

    async def _extract_images(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str,
        policy: FolderPolicy,
        document: Document,
    ) -> list[Chunk]:
        """Optionally extract image content when enabled by policy."""

        self.image_extractor.enable_ocr = bool(policy.ocr_enabled)
        self.image_extractor.enable_caption = bool(policy.caption_enabled)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return await self.pdf_image_extractor.extract_images_from_pdf(file_path, doc_id, folder_id, document.is_scanned)
        if ext == ".docx":
            return await self.docx_image_extractor.extract_images_from_docx(file_path, doc_id, folder_id)
        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return await self.image_extractor.extract_from_file(file_path, doc_id, folder_id)
        return []

    def _collect_files(self, folder_path: Path) -> list[Path]:
        """Return supported files under a folder, skipping hidden paths."""

        files: list[Path] = []
        for item in folder_path.rglob("*"):
            if not item.is_file():
                continue
            if item.name.startswith(".") or any(part.startswith(".") for part in item.parts):
                continue
            if item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(item)
        return sorted(files)
