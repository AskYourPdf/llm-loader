"""
Example usage of the llm_loader package for RAG applications.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from llm_loader.rag_processor import RAGProcessor

# Load environment variables
load_dotenv()

def main():
    # Example document path (replace with your 2k report file path)
    document_path = "path/to/your/report.pdf"
    
    # Example with OpenAI
    openai_processor = RAGProcessor(
        llm=ChatOpenAI(model_name="gpt-4"),
        save_output=True,
        output_dir="outputs/openai"
    )
    
    # Process with different chunking strategies
    # 1. Page chunking
    result_page = openai_processor.process(
        source=document_path,
        query="What are the key findings in the report?",
        chunking_strategy="page"
    )
    print("\nResults with page chunking:")
    print(result_page["response"])
    
    # 2. Contextual chunking
    result_contextual = openai_processor.process(
        source=document_path,
        query="What are the main recommendations?",
        chunking_strategy="contextual"
    )
    print("\nResults with contextual chunking:")
    print(result_contextual["response"])
    
    # 3. Custom chunking
    result_custom = openai_processor.process(
        source=document_path,
        query="What is the executive summary?",
        chunking_strategy="custom",
        custom_separators=["\n\nSection", "\n\nChapter", "\n\n", "\n"]
    )
    print("\nResults with custom chunking:")
    print(result_custom["response"])
    
    # Example with Anthropic
    anthropic_processor = RAGProcessor(
        llm=ChatAnthropic(model_name="claude-3-sonnet-20240229"),
        save_output=True,
        output_dir="outputs/anthropic"
    )
    
    result_anthropic = anthropic_processor.process(
        source=document_path,
        query="What are the financial implications discussed in the report?",
        chunking_strategy="contextual"
    )
    print("\nResults with Anthropic:")
    print(result_anthropic["response"])
    
    # Example with Google's Gemini
    gemini_processor = RAGProcessor(
        llm=ChatGoogleGenerativeAI(model="gemini-pro"),
        save_output=True,
        output_dir="outputs/gemini"
    )
    
    result_gemini = gemini_processor.process(
        source=document_path,
        query="What risks are identified in the report?",
        chunking_strategy="contextual"
    )
    print("\nResults with Gemini:")
    print(result_gemini["response"])
    
    # Example with a URL source
    url = "https://example.com/some-article"
    result_url = openai_processor.process(
        source=url,
        query="What is the main topic of this article?",
        chunking_strategy="contextual"
    )
    print("\nResults from URL:")
    print(result_url["response"])


if __name__ == "__main__":
    main() 