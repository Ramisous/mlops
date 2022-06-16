from pathlib import Path

from config import config
from tagifai import data, main

run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
artifacts = main.load_artifacts(run_id=run_id)
assert isinstance(artifacts["label_encoder"], data.LabelEncoder)

data = {
    "texts": [
        {"text": "Transfer learning with transformers for text classification."},
        {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
    ]
}


print("yes")
