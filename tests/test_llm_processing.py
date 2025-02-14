import pytest
from PIL import Image
from langchain_core.documents import Document
from unittest.mock import Mock

from smart_llm_loader.llm import LLMProcessing
from smart_llm_loader.prompts import DEFAULT_PAGE_CHUNK_PROMPT, DEFAULT_CHUNK_PROMPT


@pytest.fixture
def llm_processor(mocker):
    # Mock all validation functions
    mocker.patch('smart_llm_loader.llm.validate_environment', return_value={"keys_in_environment": True})
    mocker.patch('smart_llm_loader.llm.supports_vision', return_value=True)
    mocker.patch('smart_llm_loader.llm.check_valid_key', return_value=True)
    return LLMProcessing(model="gemini/gemini-2.0-flash")


@pytest.fixture
def sample_image():
    return Image.new('RGB', (100, 100), color='white')


def test_validate_model_valid(mocker):
    # Mock the validation functions
    mocker.patch('smart_llm_loader.llm.validate_environment', return_value={"keys_in_environment": True})
    mocker.patch('smart_llm_loader.llm.supports_vision', return_value=True)
    mocker.patch('smart_llm_loader.llm.check_valid_key', return_value=True)

    # Should not raise any exceptions
    LLMProcessing(model="gemini/gemini-2.0-flash")


def test_validate_model_missing_env_vars(mocker):
    mocker.patch('smart_llm_loader.llm.validate_environment', return_value={"keys_in_environment": False})

    with pytest.raises(ValueError, match="Missing environment variables"):
        LLMProcessing(model="gemini/gemini-2.0-flash")


def test_validate_model_unsupported_vision(mocker):
    mocker.patch('smart_llm_loader.llm.validate_environment', return_value={"keys_in_environment": True})
    mocker.patch('smart_llm_loader.llm.supports_vision', return_value=False)

    with pytest.raises(ValueError, match="not a supported vision model"):
        LLMProcessing(model="unsupported-model")


def test_get_chunk_prompt_default_page():
    prompt = LLMProcessing.get_chunk_prompt('page')
    assert prompt == DEFAULT_PAGE_CHUNK_PROMPT


def test_get_chunk_prompt_default_contextual():
    prompt = LLMProcessing.get_chunk_prompt('contextual')
    assert prompt == DEFAULT_CHUNK_PROMPT


def test_get_chunk_prompt_custom():
    custom_prompt = "Custom test prompt"
    prompt = LLMProcessing.get_chunk_prompt('custom', custom_prompt)
    assert prompt == custom_prompt


def test_get_chunk_prompt_custom_missing():
    with pytest.raises(ValueError, match="Custom prompt is not provided"):
        LLMProcessing.get_chunk_prompt('custom')


def test_get_chunk_prompt_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid chunk strategy"):
        LLMProcessing.get_chunk_prompt('invalid')


def test_prepare_llm_messages(sample_image):
    prompt = "Test prompt"
    messages = LLMProcessing.prepare_llm_messages(sample_image, prompt)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompt
    assert messages[1]["role"] == "user"
    assert len(messages[1]["content"]) == 2
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][1]["type"] == "image_url"


def test_serialize_response():
    results = [
        {
            "markdown_chunks": [
                {"content": "Test content 1", "theme": "Theme 1"},
                {"content": "Test content 2", "theme": "Theme 2"}
            ]
        }
    ]
    file_path = "test.pdf"

    documents = LLMProcessing.serialize_response(results, file_path)

    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].page_content == "Test content 1"
    assert documents[0].metadata["semantic_theme"] == "Theme 1"
    assert documents[0].metadata["source"] == file_path


@pytest.mark.asyncio
async def test_async_process_image_with_llm_success(llm_processor, sample_image, mocker):
    # Create a mock response with the correct structure
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content='{"markdown_chunks": [{"content": "Test content", "theme": "Test theme"}]}'
            )
        )
    ]
    mocker.patch('smart_llm_loader.llm.acompletion', return_value=mock_response)

    result = await llm_processor.async_process_image_with_llm(sample_image, "Test prompt")

    assert "markdown_chunks" in result
    assert len(result["markdown_chunks"]) == 1
    assert result["markdown_chunks"][0]["content"] == "Test content"


@pytest.mark.asyncio
async def test_async_process_image_with_llm_error(llm_processor, sample_image, mocker):
    mocker.patch('smart_llm_loader.llm.acompletion', side_effect=Exception("Test error"))

    result = await llm_processor.async_process_image_with_llm(sample_image, "Test prompt")

    assert "markdown_chunks" in result
    assert result["markdown_chunks"][0]["content"] is None
