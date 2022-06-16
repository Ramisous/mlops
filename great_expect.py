from pathlib import Path
import pytest
from config import config
from tagifai import main, predict, data


run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
artifacts = main.load_artifacts(run_id=run_id)
assert isinstance(artifacts["label_encoder"], data.LabelEncoder)

data = {
    "texts": [
        {"text": "Transfer learning with transformers for text classification."},
        {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
    ]
}
response = client.post("/predict", json=data)
assert response.status_code == HTTPStatus.OK
assert response.request.method == "POST"
assert len(response.json()["data"]["predictions"]) == len(data["texts"])

print('yes')