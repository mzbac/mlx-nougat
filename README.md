# MLX Nougat

MLX Nougat is a CLI tool for OCR using the Nougat model.

## Installation

1. Install ImageMagick:

   ```bash
   brew install imagemagick
   ```

2. Configure environment variables for ImageMagick:

   Add the following lines to your shell configuration file (e.g., ~/.bashrc, ~/.zshrc):

   ```bash
   export MAGICK_HOME=$(brew --prefix imagemagick)
   export PATH=$MAGICK_HOME/bin:$PATH
   export DYLD_LIBRARY_PATH=$MAGICK_HOME/lib:$DYLD_LIBRARY_PATH
   ```

   After adding these lines, reload your shell configuration or restart your terminal.

3. Install MLX Nougat:

   ```bash
   git clone git@github.com:mzbac/mlx-nougat.git
   cd mlx-nougat
   pip install .
   ```

## Usage

After installation, you can use MLX Nougat from the command line:

```bash
mlx_nougat --input <path_to_image_or_pdf_or_url> [--output <output_file>] [--model <model_name_or_path>]
```

### Arguments

- `--input`: (Required) Path to the input image or PDF file, or a URL to an image or PDF.
- `--output`: (Optional) Path to save the output text file. If not provided, the output will be printed to the console.
- `--model`: (Optional) Name or path of the Nougat model to use. Default is "facebook/nougat-small".

### Examples

1. Process a local image:

   ```bash
   mlx_nougat --input path/to/your/image.png --output results.txt
   ```

2. Process a local PDF:

   ```bash
   mlx_nougat --input path/to/your/document.pdf --output results.txt
   ```

3. Process a remote image:

   ```bash
   mlx_nougat --input https://example.com/image.jpg --output results.txt
   ```

4. Process a remote PDF:

   ```bash
   mlx_nougat --input https://example.com/document.pdf --output results.txt
   ```

5. Use a different model:

   ```bash
   mlx_nougat --input path/to/your/image.png --model facebook/nougat-base --output results.txt
   ```

## TODOs

- [ ] Support quantized model to improve the performance.

## Acknowledgements

This project is built upon several open-source projects and research works:

- [Nougat](https://github.com/facebookresearch/nougat): The original Nougat model developed by Facebook AI Research.
- [faster-nougat](https://github.com/zhuzilin/faster-nougat): An optimized implementation of Nougat, which inspired this MLX-based version.
- [MLX](https://github.com/ml-explore/mlx): The machine learning framework developed by Apple, used for efficient model inference in this project.
- [Transformers](https://github.com/huggingface/transformers): Hugging Face's state-of-the-art natural language processing library.
