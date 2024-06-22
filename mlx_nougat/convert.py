import json
from pathlib import Path
import shutil
from typing import Tuple
from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
from mlx_nougat.nougat import Nougat
from mlx_nougat.utils import get_model_path, save_config, save_weights
import copy


def quantize_model(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    quantized_config = copy.deepcopy(config)
    nn.quantize(model, q_group_size, q_bits)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config

def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    q_group_size: int = 64,
    q_bits: int = 4,
):
    print("[INFO] Loading")
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)
    model_path = get_model_path(hf_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    model = Nougat.from_pretrained(hf_path)
    weights = dict(tree_flatten(model.parameters()))

    print("[INFO] Quantizing")
    model.load_weights(list(weights.items()))
    weights, config = quantize_model(model, config, q_group_size, q_bits)


    for file in model_path.glob("*"):
        if file.suffix != ".safetensors":
            shutil.copy(file, mlx_path)
    del model
    save_weights(mlx_path, weights, donate_weights=True)

    save_config(config, config_path=mlx_path / "config.json")





