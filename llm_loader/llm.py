import asyncio
from pathlib import Path
from typing import List, Optional, Union
from base64 import b64encode
import io
from multiprocessing import cpu_count

from PIL.Image import Image
from langchain_core.documents import Document
from pdf2image import convert_from_path
from litellm import completion, validate_environment, supports_vision, check_valid_key, acompletion

from llm_loader.prompts import DEFAULT_PAGE_CHUNK_PROMPT, DEFAULT_CHUNK_PROMPT
from llm_loader.schema import OCRResponse


class ImageProcessor:
    @staticmethod
    def pdf_to_images(file_path: Optional[Union[str, Path]] = None) -> list[Image]:
        """Convert PDF pages to images all at once for better performance.

        Args:
            file_path: Path to the PDF file to convert

        Returns:
            list[Image]: List of PIL Image objects, one per PDF page
        """
        images = convert_from_path(
            file_path,
            dpi=300,
            fmt='PNG',
            size=(None, 1056),
            thread_count=cpu_count(),
            use_pdftocairo=True,
        )
        return images

    @staticmethod
    def image_to_base64(image: Image) -> str:
        """Convert an image to a base64 string.

        Args:
            image: PIL Image object to convert

        Returns:
            str: Base64 encoded string representation of the image
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        return b64encode(img_bytes).decode('utf-8')


class LLMProcessing:
    def __init__(self, model: str = "gemini/gemini-2.0-flash", **kwargs):
        self._validate_model(model, **kwargs)
        self.model = model
        self.kwargs = kwargs

    @staticmethod
    def _validate_model(model: str, **kwargs) -> None:
        """Validate that the model is properly configured for vision tasks."""
        environment = validate_environment(model=model)
        api_key = kwargs.get("api_key")

        if not environment["keys_in_environment"] and not api_key:
            raise ValueError(f"Missing environment variables for {model}: {environment}")

        if not supports_vision(model=model):
            raise ValueError(f"Model '{model}' is not a supported vision model.")

        if not check_valid_key(model=model, api_key=api_key):
            raise ValueError(f"Failed to access model '{model}'. Please check your API key and model availability.")

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
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
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
            output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Document]:
        """Process a document with LLM for OCR and chunking.

        Args:
            file_path: Path to the document to process
            chunk_strategy: Strategy for chunking ('page', 'contextual', or 'custom')
            custom_prompt: Custom prompt to use for chunking
            output_dir: Directory to save processed output

        Returns:
            List[Document]: List of processed document chunks with metadata
        """

        async def process_pdf():
            images = ImageProcessor.pdf_to_images(file_path)
            prompt = self.get_chunk_prompt(chunk_strategy, custom_prompt)
            return await asyncio.gather(*[self.async_process_image_with_llm(img, prompt) for img in images])

        results = asyncio.run(process_pdf())
        documents = self.serialize_response(list(results), file_path)
        save_output_file(documents, output_dir)
        return documents

    async def async_process_document_with_llm(
            self,
            file_path: Optional[Union[str, Path]] = None,
            chunk_strategy: str = 'page',
            custom_prompt: Optional[str] = None,
            output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Document]:
        """Process a document with LLM for OCR and chunking asynchronously."""
        images = ImageProcessor.pdf_to_images(file_path)
        prompt = self.get_chunk_prompt(chunk_strategy, custom_prompt)
        results = list(await asyncio.gather(*[self.async_process_image_with_llm(img, prompt) for img in images]))
        documents = self.serialize_response(list(results), file_path)
        save_output_file(documents, output_dir)
        return documents

    async def async_process_image_with_llm(self, page_as_image: Image, prompt: str) -> dict:
        """Convert image to base64 and chunk the image with LLM asynchronously.

        Args:
            page_as_image: PIL Image object to process
            prompt: Prompt to use for LLM processing

        Returns:
            dict: Processed chunks with content and metadata
        """
        messages = self.prepare_llm_messages(page_as_image, prompt)
        try:
            response = await acompletion(
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

    def process_image_with_llm(self, page_as_image: Image, prompt: str) -> dict:
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

    def chunk_text_with_llm(self, text: str, prompt: str) -> dict:
        """Chunk text with LLM."""
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    },
                ],
                response_format=OCRResponse,
                **self.kwargs,
            )

            result = response.choices[0].message.content
            _response = OCRResponse.parse_raw(result)
            return _response.dict()

        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return {"chunks": [{"content": None, "page": None, "theme": None}]}
