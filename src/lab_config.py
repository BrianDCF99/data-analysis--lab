#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "source_root": "../trading-data/data/bybit-perp-data/symbols",
        "parquet_root": "data/parquet/bybit_perp",
        "duckdb_path": "data/duckdb/bybit_perp.duckdb",
        "plots_root": "outputs/plots",
        "notes_root": "outputs/notes",
    },
    "convert": {
        "chunk_size": 100000,
        "compression": "zstd",
        "workers": 8,
        "overwrite": False,
        "default_all_symbols": True,
    },
    "plot": {
        "default_symbol": "AZTECUSDT",
        "start_ts_ms": None,
        "end_ts_ms": None,
    },
    "browser": {
        "host": "127.0.0.1",
        "port": 8051,
    },
    "ai": {
        "gemini_enabled": True,
        "gemini_api_key_env": "GEMINI_API_KEY",
        "gemini_model": "gemini-2.0-flash",
        "gemini_timeout_sec": 45,
        "gemini_temperature": 0.15,
        "gemini_max_output_tokens": 900,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)  # type: ignore[index]
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return DEFAULT_CONFIG
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return DEFAULT_CONFIG
    return deep_merge(DEFAULT_CONFIG, raw)


def path_from_config(config: dict[str, Any], section: str, key: str) -> Path:
    value = config.get(section, {}).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing config path: {section}.{key}")
    return Path(value)
