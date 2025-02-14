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


def save_output_file(documents: List[Document], output_dir: Path) -> None:
    """Save the chunks and input file to a folder."""
    if not output_dir or not documents:
        return

    output_dir.mkdir(exist_ok=True)
    chunks_data = [
        {
            "content": doc.page_content,
            "metadata": {**doc.metadata, "source": str(doc.metadata["source"]) if "source" in doc.metadata else None},
        }
        for doc in documents
    ]

    identifier = documents[0].metadata.get("source") or output_dir.stem
    identifier = Path(identifier).name.rsplit('.', 1)[0]

    chunks_file = output_dir / f"{identifier}_chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)


def copy_file(file_path: Path, output_file: Path) -> None:
    """Copy the file to the output directory."""
    try:
        shutil.copy2(file_path, output_file)
    except shutil.SameFileError:
        pass
