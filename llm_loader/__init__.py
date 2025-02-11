import subprocess
import sys
import platform
from pathlib import Path

def _check_poppler_installation():
    """Check if poppler is installed and provide installation instructions if it's not."""
    system = platform.system().lower()
    
    try:
        if system == "darwin":  # macOS
            subprocess.run(["pdftoppm", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        elif system == "linux":
            subprocess.run(["pdftoppm", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        elif system == "windows":
            # On Windows, check if poppler is in PATH or in common installation locations
            poppler_path = None
            for path in sys.path:
                if Path(path).joinpath("poppler/bin").exists():
                    poppler_path = path
                    break
            if not poppler_path:
                raise FileNotFoundError
    except (subprocess.SubprocessError, FileNotFoundError):
        instructions = {
            "darwin": "Install poppler using: brew install poppler",
            "linux": "Install poppler using: sudo apt-get install poppler-utils (Ubuntu/Debian) or sudo yum install poppler-utils (CentOS/RHEL)",
            "windows": "Download and install poppler from: http://blog.alivate.com.au/poppler-windows/ and add it to your PATH"
        }
        
        print(f"\n⚠️  WARNING: Required system dependency 'poppler' not found!")
        print(f"This package requires poppler for PDF processing.")
        print(f"\nTo install poppler on your system:")
        print(f"{instructions.get(system, 'Please install poppler for your operating system')}")
        print("\nContinuing without PDF processing capability...\n")

# Run the check when the package is imported
_check_poppler_installation()

# Import main package components
from .document_loader import LLMLoader

__version__ = "0.1.0"
