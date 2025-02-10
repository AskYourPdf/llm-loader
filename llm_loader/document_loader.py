"""
Document loader module for handling different types of inputs (files and URLs).
"""
from pathlib import Path
import tempfile
from typing import List, Optional, Iterator, Dict, Union
from base64 import b64encode
import io
from multiprocessing import Pool, cpu_count

from PIL.Image import Image
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from pdf2image import convert_from_path
from pydantic import BaseModel
from litellm import completion
import requests


def is_pdf(url: str, response: requests.Response) -> bool:
    """Check if the URL points to a PDF file."""
    return url.lower().endswith('.pdf') or response.headers.get('Content-Type', '').lower() in [
        'application/pdf',
        'binary/octet-stream'
    ]


DEFAULT_CHUNK_PROMPT = """OCR the following page into Markdown. Tables should be formatted as HTML.
Do not sorround your output with triple backticks.

Chunk the document into sections of roughly 250 - 1000 words. Our goal is
to identify parts of the page with same semantic theme. These chunks will
be embedded and used in a RAG pipeline.
"""

DEFAULT_PAGE_CHUNK_PROMPT = """OCR the following page into Markdown. Tables should be formatted as HTML.
Do not sorround your output with triple backticks. The contents of the page should be returned as a single chunk.

Images in the document should be properly discribed in details such that an LLM can understand the image and answer questions about the image without seeing the image.
The description should be returned as a part of the page content.
"""


class Chunk(BaseModel):
    content: str
    page: Optional[int] = None
    theme: Optional[str] = None


class OCRResponse(BaseModel):
    chunks: List[Chunk]


class DocumentLoader(BaseLoader):
    """A flexible document loader that supports multiple input types."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None, url: Optional[str] = None,
                 chunk_strategy: str = 'page', custom_prompt: Optional[str] = None, model: str = "gemini/gemini-2.0-flash", **kwargs):
        """Initialize the DocumentLoader with a file path or URL."""

        """
        Args:
            file_path: Path to the file to load
            url: URL to load the document from
            chunk_strategy: Strategy to use for chunking the document page, contextual or custom
            custom_prompt: Custom prompt to use for chunking the document, this will override the default prompt
            **kwargs: Additional arguments that will be passed to the litellm.completion method. Refer: https://docs.litellm.ai/docs/completion/input and https://docs.litellm.ai/docs/providers 
        """
        self.file_path = file_path
        self.url = url
        self.chunk_strategy = chunk_strategy
        self.custom_prompt = custom_prompt
        self.model = model
        self.kwargs = kwargs
        
        if file_path and url:
            raise ValueError("Only one of file_path or url should be provided.")

        if not file_path and not url:
            raise ValueError("Either file_path or url must be provided.")

    def get_chunk_prompt(self, strategy: str) -> str:
        if strategy == 'custom' and not self.custom_prompt:
            raise ValueError("Custom prompt is not provided. A custom prompt is required for 'custom' strategy.")

        if self.custom_prompt:
            return self.custom_prompt

        elif strategy == 'page':
            return DEFAULT_PAGE_CHUNK_PROMPT

        elif strategy == 'contextual':
            return DEFAULT_CHUNK_PROMPT

        else:
            raise ValueError(f"Invalid chunk strategy: {strategy}, must be one of 'page', 'contextual' or 'custom'")

    @staticmethod
    def _load_from_path(file_path: Union[str, Path]) -> Path:
        """Load documents from a file path."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return file_path

    @staticmethod
    def _load_from_url(url: str) -> Path:
        """Load documents from a URL."""
        response = requests.get(url)
        response.raise_for_status()
        is_link_to_pdf = is_pdf(url, response)

        if is_link_to_pdf:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            temp_path.write_bytes(response.content)
            return temp_path
        else:
            raise ValueError("The URL does not point to a PDF file.")

    async def aload(self) -> List[Document]:
        """Load data into Document objects."""
        return [document async for document in self.alazy_load()]

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        """Load Documents and split into chunks using LLM-based OCR processing."""

        images = list(self.pdf_to_images())
        prompt = self.get_chunk_prompt(self.chunk_strategy)
        with Pool(processes=min(cpu_count(), len(images))) as pool:
            results = pool.starmap(self._process_with_llm, [(img, prompt) for img in images])

        documents = []
        for page_num, result in enumerate(results):
            for chunk in result['chunks']:
                if chunk.get('theme') is None and chunk.get('content') is None:
                    continue

                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        'page': page_num,
                        'semantic_theme': chunk.get('theme'),
                        'source': self.file_path,
                    },
                )
                documents.append(doc)

        return documents

    def _process_with_llm(self, page_as_image: Image, prompt: str) -> dict:
        """Process a page with LLM for OCR and chunking."""
        img_byte_arr = io.BytesIO()
        page_as_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        base64_image = b64encode(img_bytes).decode('utf-8')
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Process this image:"},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"},
                ],
            },
        ]

        try:
            response = completion(
                model=self.model,
                messages=messages,
                response_format=OCRResponse,
                **self.kwargs,
            )

            result = response.choices[0].message.content
            _response = OCRResponse.parse_raw(result)
            print("Processed page successfully ", _response.dict())
            return _response.dict()

        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return {"chunks": [{"content": None, "page": None, "theme": None}]}
        
    def pdf_to_images(self) -> list[Image]:
        """Convert PDF pages to images all at once for better performance."""

        return convert_from_path(
            self.file_path,
            dpi=300,
            fmt='PNG',
            size=(None, 1056),
            thread_count=cpu_count(),  # Maximize thread usage
            use_pdftocairo=True,
        )

    def lazy_load(self) -> list[Image]:
        pass

    @classmethod
    def load() -> List[Document]:
        """
        Load documents from either a file path or URL.
        
        Args:
            source: File path or URL to load documents from
            
        Returns:
            List of Document objects
        """
        pass
