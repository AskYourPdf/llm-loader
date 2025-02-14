import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables needed for testing."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")


@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_pdf_path(test_dir):
    """Create a sample PDF file for testing."""
    pdf_path = test_dir / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")  # Minimal valid PDF
    return pdf_path


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return Image.new('RGB', (100, 100), color='white')


@pytest.fixture
def output_dir(test_dir):
    """Create an output directory for test results."""
    output_path = test_dir / "output"
    output_path.mkdir(exist_ok=True)
    return output_path
