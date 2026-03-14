"""Folder profiling logic for domain-aware ingestion."""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from pypdf import PdfReader

from src.core.policies.folder_policy import (
    AuthorityLevel,
    DocumentType,
    EntitySchema,
    FolderPolicy,
    RetrievalStrategy,
)
from src.utils.ids import generate_folder_id

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "these",
    "those",
    "into",
    "your",
    "their",
    "they",
    "have",
    "will",
    "would",
    "shall",
    "should",
    "clinical",
    "trial",
    "trials",
    "document",
    "documents",
    "guideline",
    "guidelines",
}

_PHRASE_HINTS = (
    "estimand",
    "sensitivity analysis",
    "randomization",
    "survival analysis",
    "dose finding",
    "power analysis",
    "experimental design",
    "regulatory guidance",
    "statistical principles",
)


class FolderScanResult(BaseModel):
    """Basic inventory for a folder."""

    all_files: list[Path] = Field(default_factory=list)
    pdf_files: list[Path] = Field(default_factory=list)
    docx_files: list[Path] = Field(default_factory=list)
    tabular_files: list[Path] = Field(default_factory=list)
    image_files: list[Path] = Field(default_factory=list)
    total_size_bytes: int = 0

    @property
    def total_count(self) -> int:
        """Return the number of files in the scan."""

        return len(self.all_files)

    @property
    def has_tables(self) -> bool:
        """Return whether tabular files were detected."""

        return bool(self.tabular_files)

    @property
    def has_images(self) -> bool:
        """Return whether image files were detected."""

        return bool(self.image_files)


class FolderProfilerAgent:
    """Create deterministic folder policies from folder structure and sample text."""

    def __init__(self, max_samples: int = 3):
        self.max_samples = max_samples
        self.logger = logging.getLogger(self.__class__.__name__)

    async def profile_folder(self, folder_path: Path) -> FolderPolicy:
        """Profile a folder and return a retrieval policy."""

        folder_path = folder_path.resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")

        scan = self._scan_folder(folder_path)
        sample_texts = self._sample_text(folder_path, scan)
        combined_text = "\n".join(sample_texts)
        topics = self._extract_topics(folder_path.name, combined_text)
        document_types = self._infer_document_types(folder_path.name, scan)
        authority = self._infer_authority(folder_path.name)
        expected_entities = self._infer_entities(topics, authority)

        policy = FolderPolicy(
            folder_id=generate_folder_id(folder_path),
            folder_name=folder_path.name,
            folder_path=str(folder_path),
            document_types=document_types,
            primary_domain="clinical research",
            authority_level=authority,
            summary=self._build_summary(folder_path.name, document_types, topics),
            key_topics=topics,
            expected_entities=expected_entities,
            retrieval_strategy=self._infer_retrieval_strategy(scan, document_types),
            keyword_boost=1.15 if authority == AuthorityLevel.PRIMARY else 1.0,
            requires_exact_match=authority == AuthorityLevel.PRIMARY,
            total_documents=scan.total_count,
            has_tables=scan.has_tables,
            has_images=scan.has_images,
            average_doc_length=self._average_sample_length(sample_texts),
            access_restrictions=["citation_required"],
            citation_required=True,
            image_processing_required=scan.has_images,
            ocr_enabled=False,
            caption_enabled=False,
            has_standalone_images=scan.has_images,
            updated_at=datetime.now(tz=timezone.utc),
        )

        logger.info(
            "Profiled folder",
            extra={
                "folder": folder_path.name,
                "documents": scan.total_count,
                "topics": topics,
                "document_types": [item.value for item in document_types],
            },
        )
        return policy

    def _scan_folder(self, folder_path: Path) -> FolderScanResult:
        """Collect a typed inventory for the folder."""

        result = FolderScanResult()
        for item in sorted(folder_path.rglob("*")):
            if not item.is_file() or item.name.startswith("."):
                continue

            result.all_files.append(item)
            result.total_size_bytes += item.stat().st_size

            suffix = item.suffix.lower()
            if suffix == ".pdf":
                result.pdf_files.append(item)
            elif suffix == ".docx":
                result.docx_files.append(item)
            elif suffix in {".xlsx", ".xls", ".csv"}:
                result.tabular_files.append(item)
            elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                result.image_files.append(item)

        return result

    def _sample_text(self, folder_path: Path, scan: FolderScanResult) -> list[str]:
        """Read a small amount of text for topic inference."""

        samples: list[str] = [folder_path.name.replace("_", " ")]
        for file_path in scan.pdf_files[: self.max_samples]:
            try:
                reader = PdfReader(str(file_path))
                extracted = []
                for page in reader.pages[:2]:
                    extracted.append((page.extract_text() or "").strip())
                text = "\n".join(part for part in extracted if part)
                if text:
                    samples.append(text[:3000])
            except Exception as exc:
                self.logger.debug("Could not sample PDF", extra={"file": str(file_path), "error": str(exc)})

        return samples

    def _extract_topics(self, folder_name: str, combined_text: str) -> list[str]:
        """Extract a concise set of domain topics."""

        normalized = f"{folder_name} {combined_text}".lower()
        topics: list[str] = [phrase for phrase in _PHRASE_HINTS if phrase in normalized]

        tokens = re.findall(r"[a-z][a-z0-9_-]{2,}", normalized)
        counts = Counter(token for token in tokens if token not in _STOP_WORDS)
        for token, _ in counts.most_common(10):
            if token not in topics:
                topics.append(token)
            if len(topics) >= 8:
                break

        if not topics:
            topics = ["clinical research", "statistical methods"]
        return topics[:8]

    def _infer_document_types(self, folder_name: str, scan: FolderScanResult) -> list[DocumentType]:
        """Infer document types from the folder label and file mix."""

        folder_name_lower = folder_name.lower()
        inferred: list[DocumentType] = []
        if any(token in folder_name_lower for token in ("fda", "ema", "ich", "guideline")):
            inferred.append(DocumentType.GUIDELINE)
        if any(token in folder_name_lower for token in ("publication", "paper", "journal")):
            inferred.append(DocumentType.PUBLICATION)
        if any(token in folder_name_lower for token in ("rpackage", "package", "software")):
            inferred.append(DocumentType.SOFTWARE_REFERENCE)
        if scan.has_tables:
            inferred.append(DocumentType.DATA_TABLE)
        if not inferred:
            inferred.append(DocumentType.UNKNOWN)
        return inferred

    def _infer_authority(self, folder_name: str) -> AuthorityLevel:
        """Infer source authority from the folder name."""

        folder_name_lower = folder_name.lower()
        if any(token in folder_name_lower for token in ("fda", "ema", "ich", "guideline")):
            return AuthorityLevel.PRIMARY
        if "archive" in folder_name_lower:
            return AuthorityLevel.ARCHIVED
        return AuthorityLevel.SECONDARY

    def _infer_entities(self, topics: list[str], authority: AuthorityLevel) -> list[EntitySchema]:
        """Return domain entities relevant to the sample corpus."""

        entities = [
            EntitySchema(entity_type="regulatory_body", examples=["FDA", "EMA", "ICH"]),
            EntitySchema(entity_type="study_design", examples=["randomization", "estimand", "sensitivity analysis"]),
        ]
        if any("software" in topic or "design" in topic for topic in topics):
            entities.append(EntitySchema(entity_type="software_package", examples=["skpr", "OptimalDesign", "AlgDesign"]))
        if authority == AuthorityLevel.SECONDARY:
            entities.append(EntitySchema(entity_type="publication_type", examples=["journal article", "methods paper"]))
        return entities

    def _infer_retrieval_strategy(
        self,
        scan: FolderScanResult,
        document_types: list[DocumentType],
    ) -> RetrievalStrategy:
        """Choose a reasonable default retrieval strategy."""

        if scan.has_tables:
            return RetrievalStrategy.STRUCTURED
        if DocumentType.SOFTWARE_REFERENCE in document_types:
            return RetrievalStrategy.SPARSE
        return RetrievalStrategy.HYBRID

    def _build_summary(
        self,
        folder_name: str,
        document_types: list[DocumentType],
        topics: list[str],
    ) -> str:
        """Generate a deterministic human-readable summary."""

        type_label = ", ".join(item.value.replace("_", " ") for item in document_types)
        topic_label = ", ".join(topics[:4])
        return f"{folder_name.replace('_', ' ')} contains {type_label} material focused on {topic_label}."

    def _average_sample_length(self, sample_texts: list[str]) -> float | None:
        """Return the average sample length when available."""

        lengths = [len(sample.strip()) for sample in sample_texts if sample.strip()]
        if not lengths:
            return None
        return round(sum(lengths) / len(lengths), 2)
