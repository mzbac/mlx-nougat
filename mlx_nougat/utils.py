import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError

def custom_roll(arr, shift, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    if not isinstance(shift, tuple):
        shift = (shift,)
    
    for ax, sh in zip(axis, shift):
        arr = custom_roll_single_axis(arr, sh, ax)
    
    return arr

def custom_roll_single_axis(arr, shift, axis):
    if shift == 0:
        return arr
    
    shape = arr.shape
    n = shape[axis]
    
    shift = shift % n 
    
    indices = mx.concatenate((mx.arange(n - shift, n), mx.arange(n - shift)))
    
    return mx.take(arr, indices, axis=axis)


def window_partition(input_feature, window_size):
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.reshape(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, num_channels)
    return windows


def window_reverse(windows, window_size, height, width):
    num_channels = windows.shape[-1]
    windows = windows.reshape(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, height, width, num_channels)
    return windows

class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path

def make_shards(weights: dict, max_file_size_gb: int = 5) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards

def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))
    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)