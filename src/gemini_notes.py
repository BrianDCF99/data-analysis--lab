#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GeminiNotesConfig:
    enabled: bool = True
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-2.0-flash"
    timeout_sec: float = 45.0
    temperature: float = 0.15
    max_output_tokens: int = 900


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def gemini_notes_config_from_dict(raw: Any) -> GeminiNotesConfig:
    cfg = raw if isinstance(raw, dict) else {}
    enabled_raw = cfg.get("gemini_enabled", cfg.get("enabled", True))
    enabled = _as_bool(enabled_raw, True)

    api_key_env = str(cfg.get("gemini_api_key_env", cfg.get("api_key_env", "GEMINI_API_KEY"))).strip()
    if not api_key_env:
        api_key_env = "GEMINI_API_KEY"

    model = str(cfg.get("gemini_model", cfg.get("model", "gemini-2.0-flash"))).strip()
    if not model:
        model = "gemini-2.0-flash"

    timeout_sec = max(5.0, _as_float(cfg.get("gemini_timeout_sec", cfg.get("timeout_sec", 45.0)), 45.0))
    temperature = _as_float(cfg.get("gemini_temperature", cfg.get("temperature", 0.15)), 0.15)
    max_output_tokens = max(
        200,
        _as_int(cfg.get("gemini_max_output_tokens", cfg.get("max_output_tokens", 900)), 900),
    )

    return GeminiNotesConfig(
        enabled=enabled,
        api_key_env=api_key_env,
        model=model,
        timeout_sec=timeout_sec,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _extract_text_from_response(response_json: dict[str, Any]) -> str:
    candidates = response_json.get("candidates")
    if not isinstance(candidates, list):
        return ""
    chunks: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n\n".join(chunks).strip()


def _build_ssl_context() -> ssl.SSLContext:
    # Prefer certifi CA bundle when available (fixes common macOS venv SSL trust issues).
    try:
        import certifi  # type: ignore

        cafile = certifi.where()
        if cafile:
            return ssl.create_default_context(cafile=cafile)
    except Exception:
        pass
    return ssl.create_default_context()


def build_note_structuring_prompt(note_record: dict[str, Any]) -> str:
    symbol = str(note_record.get("symbol", "UNKNOWN")).strip() or "UNKNOWN"
    title = str(note_record.get("title", "")).strip()
    body = str(note_record.get("body", "")).strip()
    timeframe = str(note_record.get("timeframe", "")).strip() or "unknown"
    created_at = str(note_record.get("created_at", "")).strip() or "unknown"

    snapshot = note_record.get("snapshot")
    snapshot_obj = snapshot if isinstance(snapshot, dict) else {}

    visible_series: list[str] = []
    raw_visible_series = snapshot_obj.get("visible_series")
    if isinstance(raw_visible_series, list):
        for item in raw_visible_series:
            if not isinstance(item, dict):
                continue
            if not item.get("visible"):
                continue
            label = str(item.get("label", "")).strip()
            if label:
                visible_series.append(label)

    selected_metric = snapshot_obj.get("selected_metric_id")
    selected_metric_str = str(selected_metric).strip() if selected_metric is not None else ""
    if not selected_metric_str:
        selected_metric_str = "none"

    visible_series_text = ", ".join(visible_series) if visible_series else "none"

    return f"""You are helping a quant researcher turn a raw discretionary chart note into a clean research hypothesis document.

Goal:
- Keep the user's intent and trading language.
- Structure the idea into concise bullet points that can later be tested programmatically across all coins.
- Do not invent facts not present in the note.

Output format:
- Use markdown.
- Use the exact sections below.
- Keep every section as bullets only.

Sections (in this order):
1. Observation
2. Hypothesis
3. Trigger Definition Candidates
4. Measurements To Run
5. Validation Plan (All Coins)
6. Risks / Confounders
7. Next Script Tasks

Context:
- Symbol: {symbol}
- Timeframe viewed: {timeframe}
- Note timestamp (user): {created_at}
- Note title: {title}
- Active chart series at capture: {visible_series_text}
- Selected right scale metric: {selected_metric_str}

Raw user note:
\"\"\"
{body}
\"\"\"
"""


def summarize_note_with_gemini(
    note_record: dict[str, Any],
    cfg: GeminiNotesConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": cfg.enabled,
        "model": cfg.model,
    }
    if not cfg.enabled:
        result["status"] = "disabled"
        return result

    api_key = os.getenv(cfg.api_key_env, "").strip()
    if not api_key:
        result["status"] = "missing_api_key"
        result["error"] = f"Environment variable {cfg.api_key_env} is not set."
        return result

    prompt = build_note_structuring_prompt(note_record)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": cfg.temperature,
            "maxOutputTokens": cfg.max_output_tokens,
        },
    }
    request_url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent?key={api_key}"
    request_body = json.dumps(payload).encode("utf-8")
    request_obj = Request(
        request_url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        ssl_context = _build_ssl_context()
        with urlopen(request_obj, timeout=cfg.timeout_sec, context=ssl_context) as response:
            response_bytes = response.read()
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        result["status"] = "error"
        result["error"] = f"HTTP {exc.code}: {detail or exc.reason}"
        return result
    except URLError as exc:
        result["status"] = "error"
        result["error"] = f"Network error: {exc.reason}"
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Unexpected error: {exc}"
        return result

    try:
        response_json = json.loads(response_bytes.decode("utf-8"))
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Invalid JSON response: {exc}"
        return result

    summary_md = _extract_text_from_response(response_json)
    if not summary_md:
        result["status"] = "empty_response"
        result["error"] = "Gemini returned no text candidates."
        result["response_json"] = response_json
        result["prompt"] = prompt
        return result

    result["status"] = "ok"
    result["summary_markdown"] = summary_md
    result["response_json"] = response_json
    result["prompt"] = prompt
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
