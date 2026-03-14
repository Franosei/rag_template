"""Optional image extraction helpers.

The current clinical-trials sample corpus is text-centric, so these extractors
gracefully no-op unless optional imaging dependencies are installed.
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.models import Chunk


class ImageExtractor:
    """Placeholder image extractor that degrades gracefully."""

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_caption: bool = False,
        min_confidence: float = 0.7,
        llm_client=None,
    ):
        self.enable_ocr = enable_ocr
        self.enable_caption = enable_caption
        self.min_confidence = min_confidence
        self.llm_client = llm_client

    async def extract_from_file(self, image_path: Path, doc_id: str, folder_id: str) -> list[Chunk]:
        """Return no chunks when optional image tooling is unavailable."""

        return []


class PDFImageExtractor:
    """Optional PDF image extractor."""

    def __init__(self, image_processor: ImageExtractor):
        self.image_processor = image_processor

    async def extract_images_from_pdf(
        self,
        pdf_path: Path,
        doc_id: str,
        folder_id: str,
        is_scanned: bool = False,
    ) -> list[Chunk]:
        """Return no chunks until optional OCR/image dependencies are installed."""

        return []


class DOCXImageExtractor:
    """Optional DOCX image extractor."""

    def __init__(self, image_processor: ImageExtractor):
        self.image_processor = image_processor

    async def extract_images_from_docx(self, docx_path: Path, doc_id: str, folder_id: str) -> list[Chunk]:
        """Return no chunks until optional OCR/image dependencies are installed."""

        return []
