"""
Document loader module for handling different types of inputs (files and URLs).
"""
import asyncio
from pathlib import Path
import tempfile
from typing import AsyncIterator, List, Optional, Iterator, Dict, Union
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
        'binary/octet-stream',
    ]


DEFAULT_CHUNK_PROMPT = """OCR the following page into Markdown. Tables should be formatted as HTML.
Do not surround your output with triple backticks.

Chunk the document into sections of roughly 250 - 1000 words. Our goal is
to identify parts of the page with same semantic theme. These chunks will
be embedded and used in a RAG pipeline.
"""

DEFAULT_PAGE_CHUNK_PROMPT = """OCR the following page into Markdown. Tables should be formatted as HTML.
Do not surround your output with triple backticks. The contents of the page should be returned as a single chunk.

Images in the document should be properly discribed in details such that an LLM can understand the image and answer questions about the image without seeing the image.
The description should be returned as a part of the page content.
"""


class Chunk(BaseModel):
    content: str
    page: Optional[int] = None
    theme: Optional[str] = None


class OCRResponse(BaseModel):
    chunks: List[Chunk]


class ImageProcessor:
    @staticmethod
    def pdf_to_images(file_path: Optional[Union[str, Path]] = None) -> list[Image]:
        """Convert PDF pages to images all at once for better performance."""

        return convert_from_path(
            file_path,
            dpi=300,
            fmt='PNG',
            size=(None, 1056),
            thread_count=cpu_count(),  # Maximize thread usage
            use_pdftocairo=True,
        )

    @staticmethod
    def image_to_base64(image: Image) -> str:
        """Convert an image to a base64 string."""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        return b64encode(img_bytes).decode('utf-8')


class LLMProcessing:
    def __init__(self, model: str = "gemini/gemini-2.0-flash", **kwargs):
        self.model = model
        self.kwargs = kwargs

    @staticmethod
    def get_chunk_prompt(strategy: str, custom_prompt: Optional[str] = None) -> str:
        if strategy == 'custom' and not custom_prompt:
            raise ValueError("Custom prompt is not provided. A custom prompt is required for 'custom' strategy.")

        if custom_prompt:
            return custom_prompt

        elif strategy == 'page':
            return DEFAULT_PAGE_CHUNK_PROMPT

        elif strategy == 'contextual':
            return DEFAULT_CHUNK_PROMPT

        else:
            raise ValueError(f"Invalid chunk strategy: {strategy}, must be one of 'page', 'contextual' or 'custom'")

    @staticmethod
    def prepare_llm_messages(page_as_image: Image, prompt: str) -> List[dict]:
        base64_image = ImageProcessor.image_to_base64(page_as_image)
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
        return messages

    @staticmethod
    def serialize_response(results: List[dict], file_path: Optional[Union[str, Path]] = None) -> List[Document]:
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
                        'source': file_path,
                    },
                )
                documents.append(doc)

        return documents

    def process_document_with_llm(
            self,
            file_path: Optional[Union[str, Path]] = None,
            chunk_strategy: str = 'page',
            custom_prompt: Optional[str] = None,
    ) -> List[Document]:
        """Process a document with LLM for OCR and chunking."""

        images = ImageProcessor.pdf_to_images(file_path)
        prompt = self.get_chunk_prompt(chunk_strategy, custom_prompt)
        with Pool(processes=min(cpu_count(), len(images))) as pool:
            results = pool.starmap(self.process_with_llm, [(img, prompt) for img in images])

        return self.serialize_response(results, file_path)

    async def async_process_document_with_llm(
            self,
            file_path: Optional[Union[str, Path]] = None,
            chunk_strategy: str = 'page',
            custom_prompt: Optional[str] = None,
    ) -> List[Document]:
        """Process a document with LLM for OCR and chunking asynchronously."""
        images = ImageProcessor.pdf_to_images(file_path)
        prompt = self.get_chunk_prompt(chunk_strategy, custom_prompt)
        results = list(await asyncio.gather(*[self.async_process_with_llm(img, prompt) for img in images]))
        return self.serialize_response(results, file_path)

    def process_with_llm(self, page_as_image: Image, prompt: str) -> dict:
        """Convert image to base64 and chunk the image with LLM."""
        messages = self.prepare_llm_messages(page_as_image, prompt)
        try:
            response = completion(
                model=self.model,
                messages=messages,
                response_format=OCRResponse,
                **self.kwargs,
            )

            result = response.choices[0].message.content
            _response = OCRResponse.parse_raw(result)
            return _response.dict()

        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return {"chunks": [{"content": None, "page": None, "theme": None}]}

    async def async_process_with_llm(self, page_as_image: Image, prompt: str) -> dict:
        """Convert image to base64 and chunk the image with LLM asynchronously."""
        messages = self.prepare_llm_messages(page_as_image, prompt)
        try:
            response = await completion(
                model=self.model,
                messages=messages,
                response_format=OCRResponse,
                **self.kwargs,
            )

            result = response.choices[0].message.content
            _response = OCRResponse.parse_raw(result)
            return _response.dict()

        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return {"chunks": [{"content": None, "page": None, "theme": None}]}


class DocumentLoader(BaseLoader):
    """A flexible document loader that supports multiple input types."""

    def __init__(
            self,
            file_path: Optional[Union[str, Path]] = None,
            url: Optional[str] = None,
            chunk_strategy: str = 'page',
            custom_prompt: Optional[str] = None,
            model: str = "gemini/gemini-2.0-flash",
            **kwargs,
    ):
        """Initialize the DocumentLoader with a file path or URL."""

        """
        Args:
            file_path: Path to the file to load
            url: URL to load the document from
            chunk_strategy: Strategy to use for chunking the document page, contextual or custom
            custom_prompt: Custom prompt to use for chunking the document, this will override the default prompt
            **kwargs: Additional arguments that will be passed to the litellm.completion method. Refer: https://docs.litellm.ai/docs/completion/input and https://docs.litellm.ai/docs/providers
        """
        self.chunk_strategy = chunk_strategy
        self.custom_prompt = custom_prompt
        self.llm_processor = LLMProcessing(model=model, **kwargs)

        if file_path and url:
            raise ValueError("Only one of file_path or url should be provided.")

        if not file_path and not url:
            raise ValueError("Either file_path or url must be provided.")

        self.file_path = self._load_from_path(file_path) if file_path else self._load_from_url(url)

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

    async def aload(self) -> list[Document]:
        """Load Documents and split into chunks using LLM-based OCR processing. async version"""
        return await self.llm_processor.async_process_document_with_llm(self.file_path, chunk_strategy="page")

    def load(self) -> List[Document]:
        """
        Load documents from either a file path or URL.

        Args:
            source: File path or URL to load documents from

        Returns:
            List of Document objects without chunked pages
        """
        return self.llm_processor.process_document_with_llm(self.file_path, chunk_strategy="page")

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        """Load Documents and split into chunks using LLM-based OCR processing."""
        return self.llm_processor.process_document_with_llm(self.file_path, self.chunk_strategy, self.custom_prompt)

    def lazy_load(self) -> Iterator[Document]:
        """Load Documents lazily, processing and yielding one page at a time."""
        images = ImageProcessor.pdf_to_images(self.file_path)
        prompt = self.llm_processor.get_chunk_prompt('page')

        for page_num, image in enumerate(images):
            result = self.llm_processor.process_with_llm(image, prompt)
            for chunk in result['chunks']:
                if chunk.get('content') is None:
                    continue

                yield Document(
                    page_content=chunk['content'],
                    metadata={
                        'page': page_num,
                        'semantic_theme': chunk.get('theme'),
                        'source': self.file_path,
                    },
                )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load Documents lazily and asynchronously, processing and yielding one page at a time."""
        images = ImageProcessor.pdf_to_images(self.file_path)
        prompt = self.llm_processor.get_chunk_prompt('page')

        for page_num, image in enumerate(images):
            result = await self.llm_processor.async_process_with_llm(image, prompt)
            for chunk in result['chunks']:
                if chunk.get('content') is None:
                    continue

                yield Document(
                    page_content=chunk['content'],
                    metadata={
                        'page': page_num,
                        'semantic_theme': chunk.get('theme'),
                        'source': self.file_path,
                    },
                )
