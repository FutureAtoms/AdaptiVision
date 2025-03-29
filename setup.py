from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="adaptivision",
    version="0.1.0",
    author="Abhilash Chadhar",
    author_email="abhilash.chadhar@example.com",
    description="Adaptive Context-Aware Object Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhilashchadhar/AdaptiVision",
    packages=find_packages(),
    package_data={
        "adaptivision": ["*.py", "*.yaml"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "adaptivision=src.cli:main",
        ],
    },
)

# Print installation instructions
if __name__ == "__main__":
    print("""
=======================================================
AdaptiVision Installation
=======================================================

Installation is complete! To use AdaptiVision, you need to download model weights:

1. Create weights directory:
   mkdir -p weights

2. Download model weights (choose one):
   
   # macOS/Linux
   curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o weights/model_n.pt
   
   # Windows (PowerShell)
   Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -OutFile weights\\model_n.pt

3. Try it out:
   adaptivision detect --image path/to/image.jpg --weights weights/model_n.pt

For documentation, see: https://github.com/abhilashchadhar/AdaptiVision
=======================================================
""") 