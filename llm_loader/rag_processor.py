"""
RAG processor module that handles document processing, LLM integration, and output management.
"""
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from .document_loader import DocumentLoader
from .text_chunker import TextChunker


class RAGProcessor:
    """Main RAG processing class that handles the complete RAG pipeline."""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings_model: Optional[Any] = None,
        save_output: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the RAG processor.
        
        Args:
            llm: Language model to use for processing
            embeddings_model: Model to use for embeddings (defaults to OpenAI)
            save_output: Whether to save outputs to JSON
            output_dir: Directory to save outputs to
        """
        self.llm = llm
        self.embeddings_model = embeddings_model or OpenAIEmbeddings()
        self.save_output = save_output
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "outputs"
        
        if self.save_output:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_retrieval_chain(self, documents: List[Document]) -> Any:
        """Create a retrieval chain from documents."""
        # Create vector store
        vector_store = FAISS.from_documents(documents, self.embeddings_model)
        retriever = vector_store.as_retriever()
        
        # Create prompt
        prompt = PromptTemplate.from_template("""
        Answer the following question based on the provided context:
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that.
        """)
        
        # Create chains
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    
    def process(
        self,
        source: Union[str, Path],
        query: str,
        chunking_strategy: str = "contextual",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        custom_separators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a document source with RAG.
        
        Args:
            source: File path or URL to process
            query: Question to answer
            chunking_strategy: Strategy for chunking text
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            custom_separators: Custom separators for chunking
            
        Returns:
            Dictionary containing the query, response, and metadata
        """
        # Load documents
        documents = DocumentLoader.load(source)
        
        # Chunk documents
        chunked_docs = TextChunker.chunk(
            documents,
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            custom_separators=custom_separators
        )
        
        # Create and run retrieval chain
        chain = self._create_retrieval_chain(chunked_docs)
        response = chain.invoke({"question": query})
        
        # Prepare output
        output = {
            "query": query,
            "response": response["answer"],
            "source": str(source),
            "metadata": {
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_chunks": len(chunked_docs)
            }
        }
        
        # Save output if requested
        if self.save_output:
            output_path = self.output_dir / f"rag_output_{hash(str(source))}.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
        
        return output 