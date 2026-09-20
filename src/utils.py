"""Shared helpers for loading config and metadata."""

import json
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
