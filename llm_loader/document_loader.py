"""
Document loader module for handling different types of inputs (files and URLs).
"""
import os
from pathlib import Path
import tempfile
from typing import AsyncIterator, List, Optional, Iterator, Tuple, Union
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import requests

from llm_loader.llm import ImageProcessor, LLMProcessing
from llm_loader.utils import copy_file, save_output_file, is_pdf


class LLMLoader(BaseLoader):
    """A flexible document loader that supports multiple input types."""

    def __init__(
            self,
            file_path: Optional[Union[str, Path]] = None,
            url: Optional[str] = None,
            chunk_strategy: str = 'contextual',
            custom_prompt: Optional[str] = None,
            model: str = "gemini/gemini-2.0-flash",
            save_output: bool = False,
            output_dir: Optional[Union[str, Path]] = None,
            **kwargs,
    ):
        """Initialize the DocumentLoader with a file path or URL."""

        """
        Args:
            file_path: Path to the file to load
            url: URL to load the document from
            chunk_strategy: Strategy to use for chunking the document page, contextual or custom
            custom_prompt: Custom prompt to use for chunking the document, this will override the default prompt
            save_output: Whether to save the output files
            output_dir: Directory to save output files (if save_output is True)
            **kwargs: Additional arguments that will be passed to the litellm.completion method.
            Refer: https://docs.litellm.ai/docs/completion/input and https://docs.litellm.ai/docs/providers
        """
        self.chunk_strategy = chunk_strategy
        self.custom_prompt = custom_prompt
        self.llm_processor = LLMProcessing(model=model, **kwargs)

        if file_path and url:
            raise ValueError("Only one of file_path or url should be provided.")

        if not file_path and not url:
            raise ValueError("Either file_path or url must be provided.")

        self.file_path, self.output_dir = (
            self._load_from_path(file_path, save_output, output_dir)
            if file_path
            else self._load_from_url(url, save_output, output_dir)
        )

    @staticmethod
    def _load_from_path(
            file_path: Union[str, Path], save_output: bool = False, output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Optional[Path]]:
        """Load documents from a file path."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if save_output or output_dir:
            output_dir = Path(output_dir) if output_dir else Path(f"{os.getcwd()}/{file_path.stem}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / file_path.name
            copy_file(file_path, output_file)

        return file_path, output_dir

    @staticmethod
    def _load_from_url(
            url: str, save_output: bool = False, output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Optional[Path]]:
        """Load documents from a URL."""
        response = requests.get(url)
        response.raise_for_status()
        is_link_to_pdf = is_pdf(url, response)

        if is_link_to_pdf:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(response.content)

            if save_output or output_dir:
                url_filename = url.split('/')[-1] or 'output'
                url_filename = url_filename if ".pdf" in url_filename else url_filename + ".pdf"
                output_dir = Path(output_dir) if output_dir else Path(f"{os.getcwd()}/{Path(url_filename).stem}")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / url_filename
                copy_file(temp_path, output_file)

            return temp_path, output_dir
        else:
            raise ValueError("The URL does not point to a PDF file.")

    async def aload(self) -> list[Document]:
        """Load Documents and split into chunks using LLM-based OCR processing. async version"""
        return await self.llm_processor.async_process_document_with_llm(
            self.file_path, chunk_strategy="page", output_dir=self.output_dir
        )

    def load(self) -> List[Document]:
        """Load documents from either a file path or URL.

        Processes the document using LLM-based OCR with basic page-level chunking.

        Returns:
            List[Document]: List of processed document chunks
        """
        documents = self.llm_processor.process_document_with_llm(
            self.file_path, chunk_strategy="page", output_dir=self.output_dir
        )
        return documents

    def load_and_split(self, text_splitter: Optional = None) -> List[Document]:
        """Load Documents and split into chunks using LLM-based OCR processing.

        Args:
            text_splitter: Optional text splitter (not used in current implementation)

        Returns:
            List[Document]: List of processed and chunked documents based on specified strategy
        """
        documents = self.llm_processor.process_document_with_llm(
            self.file_path, self.chunk_strategy, self.custom_prompt, output_dir=self.output_dir
        )
        return documents

    def _create_document(self, chunk: dict, page_num: int) -> Document:
        """Helper method to create a Document object from a chunk."""
        return Document(
            page_content=chunk['content'],
            metadata={
                'page': page_num,
                'semantic_theme': chunk.get('theme'),
                'source': self.file_path,
            },
        )

    def lazy_load(self) -> Iterator[Document]:
        """Load Documents lazily, processing and yielding one page at a time.

        Yields:
            Document: Processed document chunks one at a time to conserve memory
        """
        images = ImageProcessor.pdf_to_images(self.file_path)
        prompt = self.llm_processor.get_chunk_prompt('page')

        documents = []
        for page_num, image in enumerate(images):
            result = self.llm_processor.process_image_with_llm(image, prompt)
            for chunk in result['markdown_chunks']:
                if chunk.get('content') is None:
                    continue
                doc = self._create_document(chunk, page_num)
                documents.append(doc)
                yield doc

        save_output_file(documents, self.output_dir)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load Documents lazily and asynchronously, processing and yielding one page at a time.

        Yields:
            Document: Processed document chunks one at a time asynchronously
        """
        images = ImageProcessor.pdf_to_images(self.file_path)
        prompt = self.llm_processor.get_chunk_prompt('page')

        documents = []
        for page_num, image in enumerate(images):
            result = await self.llm_processor.async_process_image_with_llm(image, prompt)
            for chunk in result['markdown_chunks']:
                if chunk.get('content') is None:
                    continue
                doc = self._create_document(chunk, page_num)
                documents.append(doc)
                yield doc

        save_output_file(documents, self.output_dir)
