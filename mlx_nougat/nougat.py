import glob
import json
from pathlib import Path
from huggingface_hub import snapshot_download

import mlx.core as mx
import mlx.nn as nn

from mlx_nougat.mbartModel import MBartConfig, MBartModel
from mlx_nougat.donutSwinModel import DonutSwinModel, DonutSwinModelConfig
from mlx_nougat.utils import get_model_path

class Nougat(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None
    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = get_model_path(path_or_hf_repo)

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

        model.encoder = donutSwinModel
        model.decoder = mBartModel


        if (quantization := model_config.get("quantization", None)) is not None:
            def class_predicate(p, m):
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                **quantization,
                class_predicate=class_predicate,
            )
            donutSwinModel_weights =  {k.replace("encoder.", "", 1): v for k, v in weights.items() if not k.startswith("decoder.")}
            mBartModel_weights = {k.replace("decoder.", ""): v for k, v in weights.items() if not k.startswith("encoder.")}
        else:
            donutSwinModel_weights = DonutSwinModel.sanitize(weights)
            mBartModel_weights = MBartModel.sanitize(weights)

        donutSwinModel.load_weights(list(donutSwinModel_weights.items()))
        mBartModel.load_weights(list(mBartModel_weights.items()))

        return model        
