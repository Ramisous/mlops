from pathlib import Path

from config import config
from tagifai import main

text = "detect points in image "
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
main.predict_tag(text=text, run_id=run_id)
