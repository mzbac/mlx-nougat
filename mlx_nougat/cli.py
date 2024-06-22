#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import mlx.core as mx
from PIL import Image as PILImage
from transformers import NougatProcessor
from wand.image import Image as WandImage
from wand.color import Color

from mlx_nougat.nougat import Nougat

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def extract_pdf_pages_as_images(filename: str, *, resolution=200):
    assert filename.endswith(
        ".pdf"
    ), f"{filename} does not end with .pdf, is it a pdf file?"
    print(f"start extracting {filename}")
    # Load the image using wand
    images = []
    with WandImage(filename=filename, resolution=resolution) as wand_imgs:
        print(f"the pdf has {len(wand_imgs.sequence)} pages")
        for wand_img in wand_imgs.sequence:
            wand_img.background_color = Color("white")
            wand_img.alpha_channel = "remove"
            numpy_array = np.array(wand_img)
            images.append(PILImage.fromarray(numpy_array).convert("RGB"))
    return images

def load_image(image_source):
    if image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            img = PILImage.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            )
    elif Path(image_source).is_file():
        try:
            img = PILImage.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image {image_source} with error: {e}")
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )
    
    img = img.convert('RGB')
    return img

def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits   

def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    probs = mx.softmax(logits / temperature, axis=-1)

    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))
    token = sorted_indices.squeeze(0)[sorted_token]

    return token

def sample(logits, temperature=0.0, top_p=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return top_p_sampling(logits, top_p, temperature) if top_p > 0 and top_p < 1 else mx.random.categorical(logits * (1 / temperature))
    
def generate(model, pixel_values, max_new_tokens=4096, eos_token_id=2, temperature=0.0, top_p=0.0, repetition_penalty=None, repetition_context_size=10):
    encoder_hidden_states = model.encoder(pixel_values)
    new_token = mx.array([0])
    outputs = []
    cache = None
    repetition_context = []
    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]
    
    for _ in range(max_new_tokens):
        logits, cache = model.decoder(new_token, cache, encoder_hidden_states)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
        
        new_token = sample(logits, temperature, top_p)
        
        if repetition_penalty:
            repetition_context.append(new_token.item())
        
        if new_token == eos_token_id:
            break
        outputs.append(new_token.item())
    return outputs

def main():
    parser = argparse.ArgumentParser(description="OCR tool using Nougat model")
    parser.add_argument("--model", default="facebook/nougat-small", help="Model name or path")
    parser.add_argument("--input", required=True, help="Input image or PDF file path or URL")
    parser.add_argument("--output", help="Output file path to save results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-p sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    nougat_processor = NougatProcessor.from_pretrained(args.model)
    model = Nougat.from_pretrained(args.model)

    # Handle remote PDF
    if args.input.lower().startswith(("http://", "https://")) and args.input.lower().endswith('.pdf'):
        print(f"Downloading PDF from {args.input}")
        local_filename = "downloaded_file.pdf"
        download_file(args.input, local_filename)
        args.input = local_filename

    if args.input.lower().endswith('.pdf'):
        images = extract_pdf_pages_as_images(args.input)
    else:
        images = [load_image(args.input)]

    results = []
    start_time = time.time()

    for i, img in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}")
        pixel_values = mx.array(nougat_processor(img, return_tensors="np").pixel_values).transpose(0, 2, 3, 1)
        outputs = generate(model, pixel_values, max_new_tokens=4096, 
                           eos_token_id=nougat_processor.tokenizer.eos_token_id,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty)
        results.append(nougat_processor.tokenizer.decode(outputs))

    end_time = time.time()
    elapsed_time = end_time - start_time

    output_text = "\n\n".join(results)
    print(output_text)
    print(f"\nGeneration time: {elapsed_time:.2f} seconds")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()