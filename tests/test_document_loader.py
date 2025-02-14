from pathlib import Path
import pytest
import tempfile
from unittest.mock import Mock
from langchain_core.documents import Document

from smart_llm_loader.document_loader import SmartLLMLoader


@pytest.fixture(autouse=True)
def mock_llm_validation(mocker):
    """Mock LLM validation for all tests."""
    mocker.patch('smart_llm_loader.llm.validate_environment', return_value={"keys_in_environment": True})
    mocker.patch('smart_llm_loader.llm.supports_vision', return_value=True)
    mocker.patch('smart_llm_loader.llm.check_valid_key', return_value=True)


@pytest.fixture
def sample_pdf_path(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")  # Minimal valid PDF
    return pdf_path


@pytest.fixture
def mock_response():
    mock = Mock()
    mock.content = b"%PDF-1.4\n%EOF"  # Minimal valid PDF content
    mock.headers = {"content-type": "application/pdf"}
    return mock


def test_init_with_file_path(sample_pdf_path):
    loader = SmartLLMLoader(file_path=sample_pdf_path)
    assert str(loader.file_path) == str(sample_pdf_path)
    assert loader.output_dir is None


def test_init_with_url(mocker, mock_response):
    url = "http://example.com/test.pdf"
    mocker.patch('requests.get', return_value=mock_response)

    with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
        mocker.patch('tempfile.NamedTemporaryFile', return_value=temp_file)
        loader = SmartLLMLoader(url=url)
        assert isinstance(loader.file_path, Path)


def test_init_with_both_file_and_url(sample_pdf_path):
    with pytest.raises(ValueError, match=r"Only one of file_path or url should be provided\."):
        SmartLLMLoader(file_path=sample_pdf_path, url="http://example.com/test.pdf")


def test_init_with_neither_file_nor_url():
    with pytest.raises(ValueError, match=r"Either file_path or url must be provided\."):
        SmartLLMLoader()


def test_load_from_path_with_output_dir(sample_pdf_path, tmp_path):
    output_dir = tmp_path / "output"
    loader = SmartLLMLoader(file_path=sample_pdf_path, save_output=True, output_dir=output_dir)

    assert loader.output_dir == output_dir
    assert (output_dir / sample_pdf_path.name).exists()


def test_load_from_url_invalid_content(mocker):
    url = "http://example.com/test.txt"
    mock_resp = Mock()
    mock_resp.content = b"Not a PDF"
    mock_resp.headers = {"content-type": "text/plain"}
    mocker.patch('requests.get', return_value=mock_resp)

    with pytest.raises(ValueError, match=r"The URL does not point to a PDF file\."):
        SmartLLMLoader(url=url)


def test_load_method(mocker, sample_pdf_path):
    mock_documents = [Document(page_content="Test content")]
    mocker.patch('smart_llm_loader.llm.LLMProcessing.process_document_with_llm', return_value=mock_documents)

    loader = SmartLLMLoader(file_path=sample_pdf_path)
    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].page_content == "Test content"


@pytest.mark.asyncio
async def test_aload_method(mocker, sample_pdf_path):
    mock_documents = [Document(page_content="Test content")]
    mocker.patch('smart_llm_loader.llm.LLMProcessing.async_process_document_with_llm', return_value=mock_documents)

    loader = SmartLLMLoader(file_path=sample_pdf_path)
    documents = await loader.aload()

    assert len(documents) == 1
    assert documents[0].page_content == "Test content"


def test_load_and_split_method(mocker, sample_pdf_path):
    mock_documents = [Document(page_content="Test content")]
    mocker.patch('smart_llm_loader.llm.LLMProcessing.process_document_with_llm', return_value=mock_documents)

    loader = SmartLLMLoader(file_path=sample_pdf_path, chunk_strategy="contextual")
    documents = loader.load_and_split()

    assert len(documents) == 1
    assert documents[0].page_content == "Test content"


def test_create_document(sample_pdf_path):
    loader = SmartLLMLoader(file_path=sample_pdf_path)
    chunk = {"content": "Test content", "theme": "Test theme"}
    page_num = 1

    doc = loader._create_document(chunk, page_num)

    assert isinstance(doc, Document)
    assert doc.page_content == "Test content"
    assert doc.metadata["page"] == page_num
    assert doc.metadata["semantic_theme"] == "Test theme"
    assert doc.metadata["source"] == loader.file_path


def test_lazy_load(mocker, sample_pdf_path):
    # Mock the necessary components
    mock_images = [Mock()]
    mock_result = {"markdown_chunks": [{"content": "Test content", "theme": "Test theme"}]}

    mocker.patch('smart_llm_loader.llm.ImageProcessor.pdf_to_images', return_value=mock_images)
    mocker.patch('smart_llm_loader.llm.LLMProcessing.process_image_with_llm', return_value=mock_result)

    loader = SmartLLMLoader(file_path=sample_pdf_path)
    documents = list(loader.lazy_load())

    assert len(documents) == 1
    assert documents[0].page_content == "Test content"
    assert documents[0].metadata["semantic_theme"] == "Test theme"


@pytest.mark.asyncio
async def test_alazy_load(mocker, sample_pdf_path):
    # Mock the necessary components
    mock_images = [Mock()]
    mock_result = {"markdown_chunks": [{"content": "Test content", "theme": "Test theme"}]}

    mocker.patch('smart_llm_loader.llm.ImageProcessor.pdf_to_images', return_value=mock_images)
    mocker.patch('smart_llm_loader.llm.LLMProcessing.async_process_image_with_llm', return_value=mock_result)

    loader = SmartLLMLoader(file_path=sample_pdf_path)
    documents = [doc async for doc in loader.alazy_load()]

    assert len(documents) == 1
    assert documents[0].page_content == "Test content"
    assert documents[0].metadata["semantic_theme"] == "Test theme"
