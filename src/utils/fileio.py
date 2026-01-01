from pathlib import Path
import json
import yaml
from typing import Any, Dict, List

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config file"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: Path):
    """Save data to YAML"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, path: Path, indent: int = 2):
    """Save data to JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def list_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Recursively list files with given extensions"""
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)