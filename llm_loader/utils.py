from pathlib import Path
import shutil
from typing import List

import requests
from langchain_core.documents import Document
import json


def is_pdf(url: str, response: requests.Response) -> bool:
    """Check if the URL points to a PDF file."""
    return url.lower().endswith('.pdf') or response.headers.get('Content-Type', '').lower() in [
        'application/pdf',
        'binary/octet-stream',
    ]


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    for parent in [current_file, *current_file.parents]:
        if any((parent / f).exists() for f in ['pyproject.toml', 'setup.py', '.git', 'requirements.txt', 'README.md']):
            return parent
    return Path.cwd()


def save_output_file(documents: List[Document], output_dir: Path) -> None:
    """Save the chunks and input file to a folder."""
    if not output_dir:
        return

    if not output_dir.is_absolute():
        output_dir = get_project_root() / output_dir

    output_dir.mkdir(exist_ok=True)
    chunks_data = [
        {
            "content": doc.page_content,
            "metadata": {**doc.metadata, "source": str(doc.metadata["source"]) if "source" in doc.metadata else None},
        }
        for doc in documents
    ]

    chunks_file = output_dir / f"{output_dir.stem}_chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)


def copy_file(file_path: Path, output_file: Path) -> None:
    """Copy the file to the output directory."""
    try:
        shutil.copy2(file_path, output_file)
    except shutil.SameFileError:
        pass
