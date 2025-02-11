# LLMLoader

A flexible RAG (Retrieval-Augmented Generation) package supporting multiple LLMs and document types.

## Features

- Support for multiple LLM providers
- Document loading and processing capabilities
- RAG (Retrieval-Augmented Generation) implementation
- Flexible document type support

## System Requirements

The package will automatically check for required system dependencies on import. For PDF processing functionality, you need:

- **Poppler**: The package will check if it's installed and provide installation instructions if it's missing.
  - On macOS: `brew install poppler`
  - On Ubuntu/Debian: `apt-get install poppler-utils`
  - On CentOS/RHEL: `yum install poppler-utils`
  - On Windows: Download and install from [poppler releases](http://blog.alivate.com.au/poppler-windows/)

If poppler is not found, the package will still work but PDF processing capabilities will be limited.

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

# Use the processed documents for RAG
results = loader.process(documents)
```

## Documentation

For detailed documentation and examples, please visit our [examples](./examples) directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- David Emmanuel ([@drmingler](https://github.com/drmingler))