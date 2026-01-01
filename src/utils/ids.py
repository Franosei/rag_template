import hashlib
from pathlib import Path
from typing import Union

def generate_folder_id(folder_path: Union[str, Path]) -> str:
    """Generate stable ID for a folder based on its path"""
    path_str = str(Path(folder_path).resolve())
    return f"folder_{hashlib.md5(path_str.encode()).hexdigest()[:12]}"

def generate_document_id(file_path: Union[str, Path]) -> str:
    """Generate stable ID for a document"""
    path_str = str(Path(file_path).resolve())
    return f"doc_{hashlib.md5(path_str.encode()).hexdigest()[:12]}"

def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate ID for a chunk within a document"""
    return f"{doc_id}_chunk_{chunk_index:04d}"