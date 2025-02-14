"""
Example usage of different document loaders (smart-llm-loader and PyMuPDF) for RAG applications.
"""
import os
from dotenv import load_dotenv

from smart_llm_loader import SmartLLMLoader

# Load environment variables
load_dotenv()

# OpenAI API key since we are using the gpt-4o-mini model for question-answering
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Gemini API key since we are using the gemini flash model
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"


def process_with_llmloader():
    """Process documents using SmartLLMLoader with Gemini Flash."""

    # Initialize the loader from the smart-llm-loader package
    loader = SmartLLMLoader(
        file_path="./data/test_ocr_doc.pdf",
        chunk_strategy="contextual",
        model="gemini/gemini-1.5-flash",
        save_output=True,
        # output_dir="./data",
    )

    docs = loader.load_and_split()
    return docs


def process_with_pymupdf():
    """Process documents using PyMuPDF loader."""
    import json
    from langchain_community.document_loaders import PyMuPDFLoader
    
    # Initialize the PyMuPDF loader
    loader = PyMuPDFLoader("./data/test_ocr_doc.pdf")
    docs = loader.load()

    output_data = []
    for doc in docs:
        output_data.append({
            "page": doc.metadata["page"],
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    
    # Save as JSON
    output_path = "data/pymupdf_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return docs


def main():
    results = process_with_llmloader()
    print(results)


if __name__ == "__main__":
    main()
