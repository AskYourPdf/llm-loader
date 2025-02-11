import platform
import subprocess
import sys

def check_poppler_installed():
    try:
        result = subprocess.run(["pdftoppm", "-h"], capture_output=True)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def install_poppler():
    system = platform.system().lower()
    if check_poppler_installed():
        return
    
    try:
        if system == "darwin":
            try:
                subprocess.run(["brew", "--version"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("Installing Homebrew...")
                subprocess.run(['/bin/bash', '-c', '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'])
            
            print("Installing poppler using Homebrew...")
            subprocess.run(["brew", "install", "poppler"])
            
        elif system == "linux":
            print("Installing poppler-utils...")
            subprocess.run(["apt-get", "update"])
            subprocess.run(["apt-get", "install", "-y", "poppler-utils"])
        else:
            raise RuntimeError("Please install Poppler manually from https://poppler.freedesktop.org/")
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing poppler: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    install_poppler() 