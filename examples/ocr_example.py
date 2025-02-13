"""
Example usage of different document loaders (llm_loader and PyMuPDF) for RAG applications.
"""
import os
from dotenv import load_dotenv
from llm_loader.document_loader import LLMLoader

# Load environment variables
load_dotenv()

# OpenAI API key since we are using the gpt-4o-mini model for question-answering
os.environ["OPENAI_API_KEY"] = ""

# Gemini API key since we are using the gemini flash model
os.environ["GEMINI_API_KEY"] = ""


def process_with_llmloader():
    """Process documents using LLMLoader with Gemini Flash."""

    # Initialize the loader from the llm_loader package
    loader = LLMLoader(
        file_path="./data/test_ocr_doc.pdf",
        chunk_strategy="contextual",
        model="gemini/gemini-1.5-flash",
        save_output=True,
        # output_dir="./data",
    )

    docs = loader.load_and_split()
    return docs


def main():
    results = process_with_llmloader()
    print(results)


if __name__ == "__main__":
    main()
