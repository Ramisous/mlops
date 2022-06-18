import json
import random
from urllib.request import urlopen

import numpy as np


def load_json_from_url(url):
    """Load JSON data from a URL."""
    data = json.loads(urlopen(url).read())
    return data


def load_dict(filepath):
    """Load a dictionary from a JSON's filepath."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d, filepath, cls=None, sortkeys=False):
    """Save a dictionary to a specific location."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seeds(seed=42):
    """Set seed for reproducibility."""
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)


def list_to_dict(list_of_dicts, key):

    d_b = {}
    for d_a in list_of_dicts:
        d_b_key = d_a.pop(key)
        d_b[d_b_key] = d_a
    return d_b
