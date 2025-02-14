import pytest
from PIL import Image
import io
import base64

from smart_llm_loader.llm import ImageProcessor


@pytest.fixture
def sample_pdf_path(tmp_path):
    # Create a dummy PDF file for testing
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")  # Minimal valid PDF
    return pdf_path


@pytest.fixture
def sample_image():
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    return img


def test_pdf_to_images(sample_pdf_path, mocker):
    # Mock pdf2image.convert_from_path
    mock_images = [Image.new('RGB', (100, 100)) for _ in range(2)]
    mocker.patch('smart_llm_loader.llm.convert_from_path', return_value=mock_images)

    images = ImageProcessor.pdf_to_images(sample_pdf_path)

    assert len(images) == 2
    assert all(isinstance(img, Image.Image) for img in images)


def test_image_to_base64(sample_image):
    base64_str = ImageProcessor.image_to_base64(sample_image)

    # Verify it's a valid base64 string
    assert isinstance(base64_str, str)

    # Verify we can decode it back to an image
    try:
        decoded = base64.b64decode(base64_str)
        Image.open(io.BytesIO(decoded))
    except Exception as e:
        pytest.fail(f"Failed to decode base64 string: {e}")


def test_pdf_to_images_file_not_found():
    with pytest.raises(Exception):
        ImageProcessor.pdf_to_images("nonexistent.pdf")
