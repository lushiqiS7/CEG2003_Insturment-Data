"""Shared paths, configuration, and local run logging."""

import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def logger(name):
    (ROOT / "outputs").mkdir(exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "outputs" / f"{name}.log", encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            log.addHandler(handler)
    return log
