from typing import Dict, List

import numpy as np
import torch


def custom_predict(y_prob, threshold, index):
    """Custom predict function that defaults
    to an index if conditions are not met."""
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(texts: List, artifacts: Dict, device: torch.device = torch.device("cpu")) -> Dict:
    """Predict tags for an input text using the
    best model from the `best` experiment.
    Usage:
    ```python
    texts = ["Transfer learning with BERT."]
    artifacts = load_artifacts(run_id="264ac530b78c42608e5dea1086bc2c73")
    predict(texts=texts, artifacts=artifacts)
    ```
    <pre>
    [
      {
          "input_text": "Transfer learning with BERT.",
          "preprocessed_text": "transfer learning bert",
          "predicted_tags": [
            "attention",
            "language-modeling",
            "natural-language-processing",
            "transfer-learning",
            "transformers"
          ]
      }
    ]
    </pre>
    Note:
        The input parameter `texts` can hold multiple input texts and so the resulting prediction dictionary will have `len(texts)` items.
    Args:
        texts (List): List of input texts to predict tags for.
        artifacts (Dict): Artifacts needed for inference.
        device (torch.device): Device to run model on. Defaults to CPU.
    Returns:
        Predicted tags for each of the input texts.
    """
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"],
    )
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tags": tags[i],
        }
        for i in range(len(tags))
    ]
    return predictions
