"""Prompt loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)


class PromptManager:
    """Load prompt templates from disk with lightweight caching."""

    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or Path("configs/prompts")
        self._cache: dict[str, str] = {}

    def load_prompt(self, prompt_name: str) -> str:
        """Load a prompt file by stem name."""

        if prompt_name in self._cache:
            return self._cache[prompt_name]

        prompt_path = self.prompts_dir / f"{prompt_name}.md"
        if not prompt_path.exists():
            logger.debug("Prompt file not found", extra={"prompt": prompt_name})
            return ""

        content = prompt_path.read_text(encoding="utf-8")
        self._cache[prompt_name] = content
        return content

    def render_prompt(self, prompt_name: str, variables: dict[str, object]) -> str:
        """Render a prompt using safe template substitution."""

        template = Template(self.load_prompt(prompt_name))
        return template.safe_substitute({key: str(value) for key, value in variables.items()})


prompt_manager = PromptManager()
