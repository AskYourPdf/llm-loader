"""
Text chunking module that provides different strategies for chunking documents.
"""
from typing import List, Optional, Union, Callable
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


class TextChunker:
    """Text chunking class that provides different chunking strategies."""
    
    @staticmethod
    def page_chunking(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Chunk documents by page boundaries.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunked documents
        """
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_documents(documents)
    
    @staticmethod
    def contextual_chunking(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Chunk documents using recursive character splitting for better context preservation.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunked documents
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return splitter.split_documents(documents)
    
    @staticmethod
    def custom_chunking(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ) -> List[Document]:
        """
        Chunk documents using custom separators.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separator strings to use for chunking
            
        Returns:
            List of chunked documents
        """
        if separators is None:
            separators = ["\n\n", "\n", ".", " ", ""]
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
        return splitter.split_documents(documents)
    
    @classmethod
    def chunk(
        cls,
        documents: List[Document],
        strategy: str = "contextual",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        custom_separators: List[str] = None
    ) -> List[Document]:
        """
        Main chunking method that applies the specified chunking strategy.
        
        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy to use ('page', 'contextual', or 'custom')
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            custom_separators: List of custom separators for custom chunking strategy
            
        Returns:
            List of chunked documents
        """
        if strategy == "page":
            return cls.page_chunking(documents, chunk_size, chunk_overlap)
        elif strategy == "contextual":
            return cls.contextual_chunking(documents, chunk_size, chunk_overlap)
        elif strategy == "custom":
            return cls.custom_chunking(documents, chunk_size, chunk_overlap, custom_separators)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}") 