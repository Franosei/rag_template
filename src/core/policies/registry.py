"""Persistence for folder policies."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.policies.folder_policy import FolderPolicy
from src.utils.fileio import load_json, save_json

logger = logging.getLogger(__name__)


class FolderPolicyRegistry:
    """Simple JSON-backed registry for folder policies."""

    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def upsert(self, policy: FolderPolicy) -> FolderPolicy:
        """Create or replace a folder policy on disk."""

        path = self.registry_dir / f"{policy.folder_id}.json"
        save_json(policy.model_dump(mode="json"), path)
        return policy

    def get(self, folder_id: str) -> FolderPolicy | None:
        """Fetch a folder policy by identifier."""

        path = self.registry_dir / f"{folder_id}.json"
        if not path.exists():
            return None
        try:
            return FolderPolicy.model_validate(load_json(path))
        except Exception:
            logger.warning("Skipping unreadable folder policy", extra={"path": str(path)})
            return None

    def list(self) -> list[FolderPolicy]:
        """Return all registered folder policies."""

        policies: list[FolderPolicy] = []
        for path in sorted(self.registry_dir.glob("*.json")):
            try:
                policies.append(FolderPolicy.model_validate(load_json(path)))
            except Exception:
                logger.warning("Skipping unreadable folder policy", extra={"path": str(path)})
        return policies
