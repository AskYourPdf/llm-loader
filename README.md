# LLM Loader

A flexible Python package for Retrieval-Augmented Generation (RAG) that supports multiple LLMs, document types, and chunking strategies.

## Features

- 📄 Support for multiple document types (PDF, TXT, MD) and URLs
- 🤖 Integration with multiple LLM providers (OpenAI, Anthropic, Google)
- 🔄 Multiple text chunking strategies:
  - Page-based chunking
  - Contextual chunking
  - Custom chunking with user-defined separators
- 💾 Optional JSON output saving
- 📊 Returns results in LangChain document format
- 🔍 Built-in document splitting with page number tracking

## Installation

```bash
pip install llmloader
```

## Quick Start

```python
from llm_loader.rag_processor import RAGProcessor
from langchain_openai import ChatOpenAI

# Initialize the processor
processor = RAGProcessor(
    llm=ChatOpenAI(model_name="gpt-4"),
    save_output=True
)

# Process a document
result = processor.process(
    source="path/to/document.pdf",
    query="What are the key findings?",
    chunking_strategy="contextual"
)

print(result["response"])
```

## Chunking Strategies

### Page Chunking
```python
result = processor.process(
    source="document.pdf",
    query="What are the key findings?",
    chunking_strategy="page"
)
```

### Contextual Chunking
```python
result = processor.process(
    source="document.pdf",
    query="What are the key findings?",
    chunking_strategy="contextual"
)
```

### Custom Chunking
```python
result = processor.process(
    source="document.pdf",
    query="What are the key findings?",
    chunking_strategy="custom",
    custom_separators=["\n\nSection", "\n\nChapter", "\n\n", "\n"]
)
```

## Multiple LLM Support

### OpenAI
```python
from langchain_openai import ChatOpenAI

processor = RAGProcessor(
    llm=ChatOpenAI(model_name="gpt-4")
)
```

### Anthropic
```python
from langchain_anthropic import ChatAnthropic

processor = RAGProcessor(
    llm=ChatAnthropic(model_name="claude-3-sonnet-20240229")
)
```

### Google Gemini
```python
from langchain_google_genai import ChatGoogleGenerativeAI

processor = RAGProcessor(
    llm=ChatGoogleGenerativeAI(model="gemini-pro")
)
```

## URL Support

```python
result = processor.process(
    source="https://example.com/article",
    query="What is the main topic?",
    chunking_strategy="contextual"
)
```

## Output Format

The processor returns a dictionary containing:
- Query
- Response
- Source information
- Metadata (chunking strategy, chunk size, etc.)

Example output:
```python
{
    "query": "What are the key findings?",
    "response": "The key findings are...",
    "source": "document.pdf",
    "metadata": {
        "chunking_strategy": "contextual",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "num_chunks": 15
    }
}
```

## Environment Variables

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## Examples

Check the `examples` directory for complete usage examples with different configurations and LLMs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 