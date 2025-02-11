# LLMLoader

A flexible RAG (Retrieval-Augmented Generation) package supporting multiple LLMs and document types.

## Features

- Support for multiple LLM providers
- Document loading and processing capabilities
- RAG (Retrieval-Augmented Generation) implementation
- Flexible document type support

## Installation

You can install LLMLoader using pip:

```bash
pip install llmloader
```

Or using Poetry:

```bash
poetry add llmloader
```

## Quick Start

```python
from llm_loader import LLMLoader

# Initialize the document loader
loader = LLMLoader()

# Load and process documents
documents = loader.load_documents("path/to/your/documents")
```

## Documentation

For detailed documentation and examples, please visit our [examples](./examples) directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- David Emmanuel ([@drmingler](https://github.com/drmingler))