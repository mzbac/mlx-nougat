from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlx_nougat",
    version="0.1.0",
    author="anchen",
    author_email="li.anchen.au@gmail.com",
    description="A CLI tool for OCR using the Nougat model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mzbac/mlx_nougat",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "requests",
        "pillow",
        "transformers",
        "wand",
        "mlx",
    ],
    entry_points={
        "console_scripts": [
            "mlx_nougat=mlx_nougat.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)