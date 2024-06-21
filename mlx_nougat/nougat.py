import glob
import json
from pathlib import Path
from huggingface_hub import snapshot_download

import mlx.core as mx
import mlx.nn as nn

from mlx_nougat.mbartModel import MBartConfig, MBartModel
from mlx_nougat.donutSwinModel import DonutSwinModel, DonutSwinModelConfig

class Nougat():
    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
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

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        model = Nougat()
        encoder_config = DonutSwinModelConfig.from_dict(model_config['encoder'])
        deconder_config = MBartConfig.from_dict(model_config['decoder'])
        donutSwinModel =DonutSwinModel(encoder_config)
        mBartModel = MBartModel(deconder_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        donutSwinModel_weights = DonutSwinModel.sanitize(weights)
        donutSwinModel.load_weights(list(donutSwinModel_weights.items()))
        donutSwinModel.set_dtype(mx.bfloat16)
        mBartModel_weights = MBartModel.sanitize(weights)
        mBartModel.load_weights(list(mBartModel_weights.items()))
        mBartModel.set_dtype(mx.bfloat16)
        model.encoder = donutSwinModel
        model.decoder = mBartModel
        return model        
