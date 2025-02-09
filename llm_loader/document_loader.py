"""
Document loader module for handling different types of inputs (files and URLs).
"""
from typing import Union, List, Optional
from pathlib import Path
import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain_core.documents import Document


class DocumentLoader:
    """A flexible document loader that supports multiple input types."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None, url: Optional[str] = None):
        self.file_path = file_path
        self.url = url

        if file_path and url:
            raise ValueError("Only one of file_path or url should be provided.")
        
        if not file_path and not url:
            raise ValueError("Either file_path or url must be provided.")
    
    @staticmethod
    def _load_from_path(file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a file path."""
        file_path = Path(file_path) 
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        elif file_path.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(str(file_path))
            return loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    @staticmethod
    def _load_from_url(url: str) -> List[Document]:
        """Load documents from a URL."""
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the content temporarily and use appropriate loader
        if url.endswith('.pdf'):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            temp_path.write_bytes(response.content)
            try:
                loader = PyPDFLoader(str(temp_path))
                return loader.load()
            finally:
                temp_path.unlink()
        else:
            # Assume HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main content and clean it
            main_content = soup.get_text()
            loader = BSHTMLLoader(str(main_content))
            return loader.load()
        
    
    @classmethod
    def load() -> List[Document]:
        """
        Load documents from either a file path or URL.
        
        Args:
            source: File path or URL to load documents from
            
        Returns:
            List of Document objects
        """
        source_str = str(source)
        if source_str.startswith(('http://', 'https://')):
            return cls._load_from_url(source_str)
        else:
            return cls._load_from_path(source_str) 