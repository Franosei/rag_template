import logging
from pathlib import Path
from typing import Dict, Any, Optional
from string import Template

from src.app.settings import settings
from src.utils.fileio import load_yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Loads and manages prompt templates from configs/prompts/.
    Supports simple {variable} substitution.
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Path("configs/prompts")
        self._cache: Dict[str, str] = {}
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template from disk.
        
        Args:
            prompt_name: Name without extension, e.g., "folder_profiler"
        
        Returns:
            Raw prompt template string
        """
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        prompt_path = self.prompts_dir / f"{prompt_name}.md"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        self._cache[prompt_name] = template
        logger.debug(f"Loaded prompt: {prompt_name}")
        
        return template
    
    def render_prompt(self, prompt_name: str, variables: Dict[str, Any]) -> str:
        """
        Load and render a prompt with variable substitution.
        
        Args:
            prompt_name: Name of the prompt template
            variables: Dict of variables to substitute
        
        Returns:
            Rendered prompt string
        """
        template_str = self.load_prompt(prompt_name)
        
        # Use Template for safe substitution
        template = Template(template_str)
        
        try:
            rendered = template.substitute(variables)
        except KeyError as e:
            raise ValueError(f"Missing variable in prompt '{prompt_name}': {e}") from e
        
        logger.debug(
            f"Rendered prompt: {prompt_name}",
            extra={"variables": list(variables.keys())}
        )
        
        return rendered
    
    def get_system_prompt(self, prompt_name: str) -> str:
        """
        Load a system-level prompt (no variable substitution).
        Useful for agent personalities/instructions.
        """
        return self.load_prompt(prompt_name)


# Singleton instance
prompt_manager = PromptManager()