#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from flask import Flask, jsonify, request, Response

from gemini_notes import GeminiNotesConfig, gemini_notes_config_from_dict, summarize_note_with_gemini
from lab_config import load_config, path_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local symbol browser with dropdown + prev/next controls."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Config file path (default: config.json).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="DuckDB path (overrides config).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host bind address (overrides config).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port (overrides config).",
    )
    return parser.parse_args()


def to_utc_iso_series(series: pd.Series) -> list[str]:
    ts = pd.to_datetime(series, unit="ms", utc=True)
    return [x.isoformat() for x in ts]


def to_json_safe_list(series: pd.Series) -> list[Any]:
    out: list[Any] = []
    for value in series.tolist():
        if value is None:
            out.append(None)
            continue
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                out.append(None)
            else:
                out.append(value)
            continue
        out.append(value)
    return out


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return
    try:
        raw = dotenv_path.read_text(encoding="utf-8")
    except Exception:
        return

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def create_app(
    db_path: Path,
    start_ts_ms: int | None,
    end_ts_ms: int | None,
    notes_root: Path,
    gemini_cfg: GeminiNotesConfig,
) -> Flask:
    app = Flask(__name__)

    def get_conn() -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(db_path), read_only=True)

    @app.get("/health")
    def health() -> tuple[dict[str, Any], int]:
        return {"ok": True}, 200

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    @app.get("/api/symbols")
    def api_symbols() -> tuple[dict[str, Any], int]:
        conn = get_conn()
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT symbol
                FROM symbol_timeseries_1m
                WHERE symbol IS NOT NULL AND symbol <> ''
                ORDER BY symbol
                """
            ).fetchall()
            symbols = [str(r[0]) for r in rows if r and r[0]]
            return {"symbols": symbols}, 200
        finally:
            conn.close()

    @app.get("/api/series")
    def api_series() -> tuple[dict[str, Any], int]:
        symbol = str(request.args.get("symbol", "")).upper().strip()
        if not symbol:
            return {"error": "missing symbol query param"}, 400

        conn = get_conn()
        try:
            df = conn.execute(
                """
                SELECT
                  symbol,
                  ts_ms,
                  kline_open,
                  kline_high,
                  kline_low,
                  kline_close,
                  mark_close,
                  index_close,
                  premium_close,
                  kline_volume,
                  kline_turnover,
                  open_interest,
                  buy_ratio,
                  sell_ratio,
                  long_short_ratio,
                  basis_bps
                FROM symbol_timeseries_1m
                WHERE symbol = ?
                  AND (? IS NULL OR ts_ms >= ?)
                  AND (? IS NULL OR ts_ms <= ?)
                ORDER BY ts_ms
                """,
                [symbol, start_ts_ms, start_ts_ms, end_ts_ms, end_ts_ms],
            ).fetchdf()
            if df.empty:
                return {"symbol": symbol, "rows": 0, "data": {}}, 200

            funding_df = conn.execute(
                """
                SELECT
                  ts_ms,
                  funding_rate
                FROM funding_history
                WHERE COALESCE(symbol, symbol_partition) = ?
                  AND (? IS NULL OR ts_ms >= ?)
                  AND (? IS NULL OR ts_ms <= ?)
                ORDER BY ts_ms
                """,
                [symbol, start_ts_ms, start_ts_ms, end_ts_ms, end_ts_ms],
            ).fetchdf()
        finally:
            conn.close()

        data: dict[str, Any] = {
            "ts": to_utc_iso_series(df["ts_ms"]),
            "kline_open": to_json_safe_list(df["kline_open"]),
            "kline_high": to_json_safe_list(df["kline_high"]),
            "kline_low": to_json_safe_list(df["kline_low"]),
            "kline_close": to_json_safe_list(df["kline_close"]),
            "mark_close": to_json_safe_list(df["mark_close"]),
            "index_close": to_json_safe_list(df["index_close"]),
            "premium_close": to_json_safe_list(df["premium_close"]),
            "kline_volume": to_json_safe_list(df["kline_volume"]),
            "kline_turnover": to_json_safe_list(df["kline_turnover"]),
            "open_interest": to_json_safe_list(df["open_interest"]),
            "buy_ratio": to_json_safe_list(df["buy_ratio"]),
            "sell_ratio": to_json_safe_list(df["sell_ratio"]),
            "long_short_ratio": to_json_safe_list(df["long_short_ratio"]),
            "basis_bps": to_json_safe_list(df["basis_bps"]),
            "funding_ts": [],
            "funding_rate": [],
        }
        if not funding_df.empty:
            data["funding_ts"] = to_utc_iso_series(funding_df["ts_ms"])
            data["funding_rate"] = to_json_safe_list(funding_df["funding_rate"])

        return {"symbol": symbol, "rows": int(len(df)), "data": data}, 200

    def safe_component(value: str, fallback: str = "note") -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        slug = slug.strip("._-")
        if not slug:
            return fallback
        return slug[:120]

    def read_note_record(note_path: Path) -> dict[str, Any] | None:
        try:
            raw = json.loads(note_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        return raw

    def read_optional_text(path: Path) -> str:
        try:
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
        return ""

    @app.get("/api/notes/snapshots")
    def api_notes_snapshots() -> tuple[dict[str, Any], int]:
        symbol = str(request.args.get("symbol", "")).upper().strip()
        if not symbol:
            return {"error": "missing symbol query param"}, 400

        symbol_dir = notes_root / safe_component(symbol, "UNKNOWN")
        if not symbol_dir.exists() or not symbol_dir.is_dir():
            return {"symbol": symbol, "snapshots": []}, 200

        snapshots: list[dict[str, Any]] = []
        for entry in symbol_dir.iterdir():
            if not entry.is_dir():
                continue
            note_path = entry / "note.json"
            if not note_path.exists():
                continue
            note_record = read_note_record(note_path)
            if not note_record:
                continue
            saved_at = str(note_record.get("saved_at", "")).strip()
            created_at = str(note_record.get("created_at", "")).strip()
            title = str(note_record.get("title", "")).strip()
            timeframe = str(note_record.get("timeframe", "")).strip()
            label_parts = []
            if created_at:
                label_parts.append(created_at)
            elif saved_at:
                label_parts.append(saved_at)
            if title:
                label_parts.append(title)
            label = " | ".join(label_parts) if label_parts else entry.name
            snapshots.append(
                {
                    "id": entry.name,
                    "label": label,
                    "title": title,
                    "saved_at": saved_at or None,
                    "created_at": created_at or None,
                    "timeframe": timeframe or None,
                    "ai_status": (
                        str((note_record.get("ai_summary") or {}).get("status", "")).strip()
                        if isinstance(note_record.get("ai_summary"), dict)
                        else None
                    ),
                }
            )

        snapshots.sort(key=lambda x: str(x.get("saved_at") or x.get("created_at") or x.get("id")), reverse=True)
        return {"symbol": symbol, "snapshots": snapshots}, 200

    @app.get("/api/notes/snapshot")
    def api_notes_snapshot() -> tuple[dict[str, Any], int]:
        symbol = str(request.args.get("symbol", "")).upper().strip()
        snapshot_id = str(request.args.get("snapshot_id", "")).strip()
        if not symbol:
            return {"error": "missing symbol query param"}, 400
        if not snapshot_id:
            return {"error": "missing snapshot_id query param"}, 400
        safe_snapshot_id = safe_component(snapshot_id, "note")
        if safe_snapshot_id != snapshot_id:
            return {"error": "invalid snapshot_id"}, 400

        symbol_dir = notes_root / safe_component(symbol, "UNKNOWN")
        note_dir = symbol_dir / snapshot_id
        note_path = note_dir / "note.json"
        if not note_path.exists():
            return {"error": "snapshot not found"}, 404

        note_record = read_note_record(note_path)
        if not note_record:
            return {"error": "invalid snapshot payload"}, 500

        ai_summary_obj = note_record.get("ai_summary")
        ai_summary_markdown = ""
        if isinstance(ai_summary_obj, dict):
            summary_file_raw = ai_summary_obj.get("summary_file")
            if isinstance(summary_file_raw, str) and summary_file_raw.strip():
                summary_path = Path(summary_file_raw)
                if not summary_path.is_absolute():
                    summary_path = note_dir / summary_path
                ai_summary_markdown = read_optional_text(summary_path)
        if not ai_summary_markdown:
            ai_summary_markdown = read_optional_text(note_dir / "gemini_summary.md")

        return {
            "ok": True,
            "symbol": symbol,
            "snapshot_id": snapshot_id,
            "note": note_record,
            "ai_summary_markdown": ai_summary_markdown,
        }, 200

    def parse_note_payload(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(payload, dict):
            return None, "invalid JSON payload"

        symbol = str(payload.get("symbol", "")).upper().strip()
        if not symbol:
            return None, "missing symbol"

        title = str(payload.get("title", "")).strip()
        body = str(payload.get("body", "")).strip()
        timeframe = str(payload.get("timeframe", "")).strip()
        if not title:
            return None, "missing title"
        if not body:
            return None, "missing body"

        created_at = str(payload.get("created_at", "")).strip()
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()

        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            snapshot = {}

        return {
            "symbol": symbol,
            "title": title,
            "body": body,
            "timeframe": timeframe or None,
            "created_at": created_at,
            "snapshot": snapshot,
        }, None

    @app.post("/api/notes/preview")
    def api_notes_preview() -> tuple[dict[str, Any], int]:
        payload = request.get_json(silent=True)
        note_input, error = parse_note_payload(payload)
        if error:
            return {"error": error}, 400
        assert note_input is not None

        now_utc = datetime.now(timezone.utc)
        note_record: dict[str, Any] = {
            "saved_at": now_utc.isoformat(),
            "symbol": note_input["symbol"],
            "title": note_input["title"],
            "body": note_input["body"],
            "created_at": note_input["created_at"],
            "timeframe": note_input["timeframe"],
            "snapshot": note_input["snapshot"],
            "version": 3,
            "preview": True,
        }

        try:
            ai_preview = summarize_note_with_gemini(note_record, gemini_cfg)
        except Exception as exc:
            ai_preview = {
                "enabled": gemini_cfg.enabled,
                "model": gemini_cfg.model,
                "status": "error",
                "error": f"Unexpected preview failure: {exc}",
            }

        return {"ok": True, "ai_preview": ai_preview}, 200

    @app.post("/api/notes")
    def api_notes_create() -> tuple[dict[str, Any], int]:
        payload = request.get_json(silent=True)
        note_input, error = parse_note_payload(payload)
        if error:
            return {"error": error}, 400
        assert note_input is not None
        assert isinstance(payload, dict)
        symbol = str(note_input["symbol"])
        title = str(note_input["title"])
        body = str(note_input["body"])
        timeframe = note_input["timeframe"]
        created_at = str(note_input["created_at"])
        snapshot = note_input["snapshot"]

        notes_root.mkdir(parents=True, exist_ok=True)
        symbol_dir = notes_root / safe_component(symbol, "UNKNOWN")
        symbol_dir.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc)
        note_stamp = now_utc.strftime("%Y%m%dT%H%M%S%fZ")
        file_stem = f"{note_stamp}_{safe_component(title, 'note')}"
        note_dir = symbol_dir / file_stem
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / "note.json"

        note_record: dict[str, Any] = {
            "saved_at": now_utc.isoformat(),
            "symbol": symbol,
            "title": title,
            "body": body,
            "created_at": created_at,
            "timeframe": timeframe,
            "snapshot": snapshot,
            "note_dir": str(note_dir),
            "version": 3,
        }

        image_saved = False
        image_data_url = payload.get("chart_image_data_url")
        if isinstance(image_data_url, str) and image_data_url.startswith("data:image/png;base64,"):
            b64 = image_data_url.split(",", 1)[1]
            try:
                png_bytes = base64.b64decode(b64, validate=True)
                image_path = note_dir / "chart.png"
                image_path.write_bytes(png_bytes)
                note_record["chart_image_file"] = str(image_path)
                image_saved = True
            except Exception:
                note_record["chart_image_file"] = None

        ai_preview_raw = payload.get("ai_preview")
        ai_preview = ai_preview_raw if isinstance(ai_preview_raw, dict) else {}
        ai_summary_input = str(payload.get("ai_summary_markdown", "")).strip()
        skip_ai_summary = bool(payload.get("skip_ai_summary", False))

        ai_result: dict[str, Any] = {}
        ai_source = "generated_on_save"
        if skip_ai_summary:
            ai_source = "skipped_by_user"
            ai_result = {
                "enabled": bool(ai_preview.get("enabled", gemini_cfg.enabled)),
                "model": str(ai_preview.get("model", gemini_cfg.model)),
                "status": "skipped",
                "error": ai_preview.get("error"),
                "generated_at": ai_preview.get("generated_at"),
                "prompt": ai_preview.get("prompt"),
                "response_json": ai_preview.get("response_json"),
            }
        elif ai_summary_input:
            ai_source = "user_edited_preview"
            ai_result = {
                "enabled": bool(ai_preview.get("enabled", gemini_cfg.enabled)),
                "model": str(ai_preview.get("model", gemini_cfg.model)),
                "status": str(ai_preview.get("status", "edited")),
                "error": ai_preview.get("error"),
                "generated_at": ai_preview.get("generated_at"),
                "prompt": ai_preview.get("prompt"),
                "response_json": ai_preview.get("response_json"),
                "summary_markdown": ai_summary_input,
            }
        else:
            try:
                ai_result = summarize_note_with_gemini(note_record, gemini_cfg)
            except Exception as exc:
                ai_result = {
                    "enabled": gemini_cfg.enabled,
                    "model": gemini_cfg.model,
                    "status": "error",
                    "error": f"Unexpected summarizer failure: {exc}",
                }

        ai_summary_saved = False
        ai_summary_file: str | None = None
        ai_prompt_file: str | None = None
        ai_response_file: str | None = None
        ai_status = str(ai_result.get("status", "unknown"))

        prompt_text = ai_result.get("prompt")
        if isinstance(prompt_text, str) and prompt_text.strip():
            prompt_path = note_dir / "gemini_prompt.txt"
            prompt_path.write_text(prompt_text, encoding="utf-8")
            ai_prompt_file = str(prompt_path)

        response_json = ai_result.get("response_json")
        if isinstance(response_json, dict):
            response_path = note_dir / "gemini_response.json"
            response_path.write_text(json.dumps(response_json, indent=2), encoding="utf-8")
            ai_response_file = str(response_path)

        summary_md = ai_result.get("summary_markdown")
        if isinstance(summary_md, str) and summary_md.strip():
            summary_path = note_dir / "gemini_summary.md"
            summary_path.write_text(summary_md.strip() + "\n", encoding="utf-8")
            ai_summary_saved = True
            ai_summary_file = str(summary_path)

        note_record["ai_summary"] = {
            "status": ai_status,
            "enabled": bool(ai_result.get("enabled", gemini_cfg.enabled)),
            "model": str(ai_result.get("model", gemini_cfg.model)),
            "generated_at": ai_result.get("generated_at"),
            "error": ai_result.get("error"),
            "source": ai_source,
            "summary_saved": ai_summary_saved,
            "summary_file": ai_summary_file,
            "prompt_file": ai_prompt_file,
            "response_file": ai_response_file,
        }

        note_path.write_text(json.dumps(note_record, indent=2), encoding="utf-8")

        return {
            "ok": True,
            "note_dir": str(note_dir),
            "note_file": str(note_path),
            "image_saved": image_saved,
            "ai_summary": note_record["ai_summary"],
        }, 200

    @app.get("/")
    def index() -> str:
        return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Bybit Perp Symbol Browser</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: #020617;
        color: #e2e8f0;
      }
      .wrap {
        max-width: none;
        width: 100%;
        margin: 0;
        padding: 14px 16px 20px;
        box-sizing: border-box;
      }
      .controls {
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
      }
      .controls button {
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 8px 12px;
        cursor: pointer;
        font-weight: 600;
      }
      .controls button:hover {
        background: #111f37;
      }
      .controls button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }
      .controls select {
        min-width: 240px;
        max-width: 420px;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
      }
      .meta {
        margin-top: 8px;
        color: #94a3b8;
        font-size: 13px;
      }
      .main {
        margin-top: 12px;
        display: grid;
        grid-template-columns: 280px minmax(0, 1fr);
        gap: 12px;
        align-items: stretch;
      }
      .chart-wrap {
        position: relative;
        min-width: 0;
      }
      .legend-panel {
        background: #0b1220;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 10px;
        max-height: calc(100vh - 170px);
        overflow: auto;
      }
      .legend-title {
        color: #cbd5e1;
        font-weight: 700;
        font-size: 14px;
        margin: 2px 4px 10px;
      }
      .section-title {
        color: #cbd5e1;
        font-weight: 700;
        font-size: 14px;
        margin: 0;
      }
      .toggle-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 8px;
        border-radius: 8px;
        cursor: pointer;
        color: #e2e8f0;
        user-select: none;
      }
      .toggle-row:hover {
        background: #111f37;
      }
      .toggle-row.disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }
      .toggle-row.disabled:hover {
        background: transparent;
      }
      .toggle-row input {
        width: 18px;
        height: 18px;
        cursor: pointer;
      }
      .toggle-label {
        flex: 1 1 auto;
      }
      .series-style-btn {
        margin-left: auto;
        min-width: 44px;
        text-align: center;
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 7px;
        padding: 3px 8px;
        font-size: 12px;
        line-height: 1.2;
        cursor: pointer;
      }
      .series-style-btn:hover:not(:disabled) {
        background: #111f37;
      }
      .series-style-btn:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }
      .saved-snapshots {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #1e293b;
      }
      .saved-snapshots-head {
        display: flex;
        align-items: center;
      }
      #savedSnapshotSelect {
        width: 100%;
        margin-top: 8px;
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 7px 8px;
        font-size: 12px;
      }
      #savedSnapshotText {
        width: 100%;
        min-height: 180px;
        max-height: 320px;
        margin-top: 8px;
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 8px 10px;
        box-sizing: border-box;
        font-family: inherit;
        font-size: 12px;
        resize: vertical;
      }
      #savedSnapshotText.hidden {
        display: none;
      }
      .swatch {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        flex: 0 0 14px;
      }
      .metric-axis-control {
        position: absolute;
        top: 10px;
        right: 12px;
        z-index: 30;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 8px;
        border: 1px solid #334155;
        border-radius: 8px;
        background: rgba(11, 18, 32, 0.9);
        backdrop-filter: blur(2px);
      }
      .metric-axis-control label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
      }
      .metric-axis-control select {
        min-width: 180px;
        max-width: 240px;
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
      }
      /* Keep legacy fixed hover panel hidden if cached HTML still injects it. */
      .fixed-hover-box {
        display: none !important;
      }
      .notes-modal-backdrop {
        position: fixed;
        inset: 0;
        background: rgba(2, 6, 23, 0.72);
        backdrop-filter: blur(2px);
        z-index: 90;
        display: none;
        align-items: center;
        justify-content: center;
        padding: 16px;
        box-sizing: border-box;
      }
      .notes-modal-backdrop.open {
        display: flex;
      }
      .notes-modal {
        width: min(760px, 96vw);
        max-height: 90vh;
        overflow: auto;
        background: #0b1220;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px;
      }
      .notes-modal h3 {
        margin: 0 0 10px;
      }
      .note-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .note-field {
        display: flex;
        flex-direction: column;
        gap: 5px;
      }
      .note-field label {
        font-size: 12px;
        color: #94a3b8;
      }
      .note-field input,
      .note-field textarea {
        border: 1px solid #334155;
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 8px 10px;
        font-family: inherit;
        font-size: 13px;
      }
      .note-field textarea {
        min-height: 190px;
        resize: vertical;
      }
      #noteAiSummaryInput {
        min-height: 220px;
      }
      .note-field.full {
        grid-column: 1 / -1;
      }
      .note-actions {
        margin-top: 10px;
        display: flex;
        justify-content: flex-end;
        gap: 8px;
      }
      .note-actions .hidden {
        display: none;
      }
      .note-status {
        margin-top: 8px;
        font-size: 12px;
        color: #94a3b8;
      }
      .indicator-tooltip {
        position: fixed;
        z-index: 120;
        max-width: 360px;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid #334155;
        background: rgba(15, 23, 42, 0.96);
        color: #e2e8f0;
        font-size: 12px;
        line-height: 1.4;
        box-shadow: 0 6px 18px rgba(2, 6, 23, 0.45);
        pointer-events: none;
        display: none;
      }
      #chart {
        width: 100%;
        min-width: 0;
        height: calc(100dvh - 170px);
        min-height: 620px;
        background: #020617;
        border-radius: 12px;
        border: 1px solid #1e293b;
        overflow: visible;
      }
      @media (max-width: 1000px) {
        .main {
          grid-template-columns: 1fr;
        }
        .legend-panel {
          max-height: 240px;
        }
        #chart {
          min-height: 420px;
          height: calc(100dvh - 420px);
        }
        .metric-axis-control {
          position: static;
          margin-bottom: 8px;
          width: fit-content;
        }
        .note-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="controls">
        <button id="prevBtn" title="Previous symbol (Left Arrow)">← Prev</button>
        <select id="symbolSelect"></select>
        <select id="timeframeSelect" title="Timeframe"></select>
        <button id="addNoteBtn" title="Add a local note for current view">Add Note</button>
        <button id="nextBtn" title="Next symbol (Right Arrow)">Next →</button>
      </div>
      <div class="meta" id="metaText"></div>
      <div class="meta">Keyboard: Left/Right arrows move previous/next symbol</div>
      <div class="main">
        <aside class="legend-panel">
          <div class="legend-title">Series Toggles</div>
          <div id="legendList"></div>
          <div class="saved-snapshots">
            <div class="saved-snapshots-head">
              <div class="section-title">Saved Snapshots</div>
            </div>
            <select id="savedSnapshotSelect"></select>
            <textarea id="savedSnapshotText" class="hidden" readonly placeholder=""></textarea>
          </div>
        </aside>
        <div class="chart-wrap">
          <div class="metric-axis-control">
            <label for="metricAxisSelect">Right Scale</label>
            <select id="metricAxisSelect"></select>
          </div>
          <div id="chart"></div>
        </div>
      </div>
    </div>
    <div id="noteModalBackdrop" class="notes-modal-backdrop" aria-hidden="true">
      <div class="notes-modal" role="dialog" aria-modal="true" aria-labelledby="noteModalTitle">
        <h3 id="noteModalTitle">Add Analysis Note</h3>
        <div class="note-grid">
          <div class="note-field">
            <label for="noteSymbolInput">Ticker</label>
            <input id="noteSymbolInput" type="text" readonly>
          </div>
          <div class="note-field">
            <label for="noteDateInput">Date (UTC)</label>
            <input id="noteDateInput" type="text" readonly>
          </div>
          <div class="note-field full">
            <label for="noteTitleInput">Title</label>
            <input id="noteTitleInput" type="text" placeholder="Short note title">
          </div>
          <div class="note-field full">
            <label for="noteBodyInput">Thoughts</label>
            <textarea id="noteBodyInput" placeholder="Write your notes about this setup..."></textarea>
          </div>
          <div class="note-field full">
            <label for="noteAiSummaryInput">Gemini Draft (editable before final save)</label>
            <textarea id="noteAiSummaryInput" placeholder="Click 'Generate Gemini Draft', review response, edit if needed, then save final note."></textarea>
          </div>
        </div>
        <div class="note-status" id="noteStatus"></div>
        <div class="note-actions">
          <button id="cancelNoteBtn" type="button">Cancel</button>
          <button id="generateGeminiBtn" type="button">Generate Gemini Draft</button>
          <button id="saveAsIsBtn" type="button" class="hidden">Save As-Is</button>
          <button id="saveNoteBtn" type="button">Save Final Note</button>
        </div>
      </div>
    </div>
    <div id="indicatorTooltip" class="indicator-tooltip"></div>

    <script>
      const selectEl = document.getElementById("symbolSelect");
      const prevBtn = document.getElementById("prevBtn");
      const nextBtn = document.getElementById("nextBtn");
      const metaText = document.getElementById("metaText");
      const chartEl = document.getElementById("chart");
      const legendListEl = document.getElementById("legendList");
      const savedSnapshotSelectEl = document.getElementById("savedSnapshotSelect");
      const savedSnapshotTextEl = document.getElementById("savedSnapshotText");
      const metricAxisSelectEl = document.getElementById("metricAxisSelect");
      const timeframeSelectEl = document.getElementById("timeframeSelect");
      const addNoteBtn = document.getElementById("addNoteBtn");
      const noteModalBackdrop = document.getElementById("noteModalBackdrop");
      const noteSymbolInput = document.getElementById("noteSymbolInput");
      const noteDateInput = document.getElementById("noteDateInput");
      const noteTitleInput = document.getElementById("noteTitleInput");
      const noteBodyInput = document.getElementById("noteBodyInput");
      const noteAiSummaryInput = document.getElementById("noteAiSummaryInput");
      const noteStatus = document.getElementById("noteStatus");
      const generateGeminiBtn = document.getElementById("generateGeminiBtn");
      const saveAsIsBtn = document.getElementById("saveAsIsBtn");
      const saveNoteBtn = document.getElementById("saveNoteBtn");
      const cancelNoteBtn = document.getElementById("cancelNoteBtn");
      const indicatorTooltipEl = document.getElementById("indicatorTooltip");

      const SERIES_DEFS = [
        {
          id: "trade_candle",
          type: "candlestick",
          label: "Trade Candle",
          defaultOn: true,
          group: "price",
          color: "#22c55e",
          help: "Last-traded OHLC candles from market kline data. Green means close > open; red means close < open."
        },
        {
          id: "mark_close",
          type: "line",
          key: "mark_close",
          label: "Mark Close",
          defaultOn: true,
          group: "price",
          color: "#22c55e",
          lineWidth: 1.0,
          help: "Mark close is the fair-price used for liquidation and unrealized PnL. Rising mark means fair-price repricing up; falling means repricing down."
        },
        {
          id: "index_close",
          type: "line",
          key: "index_close",
          label: "Index Close",
          defaultOn: true,
          group: "price",
          color: "#f59e0b",
          lineWidth: 1.0,
          help: "Index close is the external spot-basket reference. Up/down mostly reflects underlying spot movement."
        },
        {
          id: "premium_close",
          type: "line",
          key: "premium_close",
          label: "Premium Close",
          defaultOn: false,
          group: "metric",
          color: "#ec4899",
          lineWidth: 1.2,
          help: "Premium close tracks perp deviation versus index. More positive means perp richer than index; more negative means perp discounted."
        },
        {
          id: "basis_bps",
          type: "line",
          key: "basis_bps",
          label: "Basis (bps)",
          defaultOn: false,
          group: "metric",
          color: "#ef4444",
          lineWidth: 1.2,
          help: "Basis in bps is (mark/index - 1) * 10,000. Higher positive values mean mark above index; negative values mean mark below index."
        },
        {
          id: "kline_volume",
          type: "line",
          key: "kline_volume",
          label: "Volume",
          defaultOn: false,
          group: "metric",
          color: "#94a3b8",
          lineWidth: 1.2,
          help: "Volume is traded contract quantity per bar. Rising volume usually means stronger participation; falling volume suggests thinner activity."
        },
        {
          id: "kline_turnover",
          type: "line",
          key: "kline_turnover",
          label: "Turnover",
          defaultOn: false,
          group: "metric",
          color: "#64748b",
          lineWidth: 1.2,
          help: "Turnover is traded notional value per bar. Higher turnover means larger dollar participation."
        },
        {
          id: "open_interest",
          type: "line",
          key: "open_interest",
          label: "Open Interest",
          defaultOn: false,
          group: "metric",
          color: "#a855f7",
          lineWidth: 1.2,
          help: "Open interest is total outstanding derivative positions. Rising OI means positions are being added; falling OI means positions are closing."
        },
        {
          id: "buy_ratio",
          type: "line",
          key: "buy_ratio",
          label: "Buy Ratio",
          defaultOn: false,
          group: "metric",
          color: "#16a34a",
          lineWidth: 1.2,
          help: "Buy ratio is Bybit account-ratio share of accounts net-long, not market buy flow. Rising values mean a larger long-holder crowd."
        },
        {
          id: "sell_ratio",
          type: "line",
          key: "sell_ratio",
          label: "Sell Ratio",
          defaultOn: false,
          group: "metric",
          color: "#f43f5e",
          lineWidth: 1.2,
          help: "Sell ratio is Bybit account-ratio share of accounts net-short. Rising values mean a larger short-holder crowd."
        },
        {
          id: "long_short_ratio",
          type: "line",
          key: "long_short_ratio",
          label: "Long/Short Ratio",
          defaultOn: false,
          group: "metric",
          color: "#14b8a6",
          lineWidth: 1.2,
          help: "Long/Short ratio is buy_ratio / sell_ratio. Above 1 means long accounts dominate; below 1 means short accounts dominate."
        },
        {
          id: "funding_rate",
          type: "markers",
          key: "funding_rate",
          xKey: "funding_ts",
          label: "Funding Rate",
          defaultOn: false,
          group: "metric",
          color: "#e2e8f0",
          help: "Funding is the periodic transfer between longs and shorts on perps. More positive means longs pay shorts; more negative means shorts pay longs."
        }
      ];
      const TIMEFRAME_DEFS = [
        { id: "1m", label: "1m", minutes: 1 },
        { id: "5m", label: "5m", minutes: 5 },
        { id: "15m", label: "15m", minutes: 15 },
        { id: "30m", label: "30m", minutes: 30 },
        { id: "1h", label: "1h", minutes: 60 },
        { id: "2h", label: "2h", minutes: 120 },
        { id: "4h", label: "4h", minutes: 240 },
        { id: "8h", label: "8h", minutes: 480 },
        { id: "12h", label: "12h", minutes: 720 },
        { id: "1d", label: "1d", minutes: 1440 }
      ];

      let symbols = [];
      let idx = -1;
      let currentRawPayload = null;
      let currentPayload = null;
      const visibilityState = Object.fromEntries(SERIES_DEFS.map((s) => [s.id, s.defaultOn]));
      const seriesStyleState = Object.fromEntries(SERIES_DEFS.map((s) => [s.id, defaultSeriesStyle(s)]));
      let selectedMetricScaleId = "";
      let selectedTimeframeId = "1m";
      const viewStateBySymbol = {};
      let relayoutCaptureBound = false;
      let isProgrammaticLayoutChange = false;
      let lastTogglePanelSignature = "";
      let currentAiPreview = null;
      let aiPreviewAttempted = false;
      let aiPreviewFailed = false;
      const savedSnapshotItemsBySymbol = {};
      const selectedSnapshotIdBySymbol = {};
      let indicatorTooltipTimer = null;

      function currentSymbol() {
        if (idx < 0 || idx >= symbols.length) return null;
        return symbols[idx];
      }

      function setButtons() {
        prevBtn.disabled = idx <= 0;
        nextBtn.disabled = idx < 0 || idx >= symbols.length - 1;
      }

      function setMetaMessage(msg) {
        if (!metaText) return;
        metaText.textContent = String(msg || "");
      }

      function setSavedSnapshotText(text, visible = true) {
        if (!savedSnapshotTextEl) return;
        savedSnapshotTextEl.value = String(text || "");
        savedSnapshotTextEl.classList.toggle("hidden", !visible);
      }

      function hideSavedSnapshotText() {
        setSavedSnapshotText("", false);
      }

      function hideIndicatorTooltip() {
        if (!indicatorTooltipEl) return;
        indicatorTooltipEl.style.display = "none";
        indicatorTooltipEl.textContent = "";
      }

      function clearIndicatorTooltipTimer() {
        if (indicatorTooltipTimer) {
          clearTimeout(indicatorTooltipTimer);
          indicatorTooltipTimer = null;
        }
      }

      function showIndicatorTooltip(rowEl, helpText) {
        if (!indicatorTooltipEl || !rowEl || !helpText) return;
        const rect = rowEl.getBoundingClientRect();
        const top = Math.max(8, Math.floor(rect.top + window.scrollY + rect.height + 6));
        const left = Math.max(8, Math.floor(rect.left + window.scrollX + 4));
        indicatorTooltipEl.textContent = String(helpText);
        indicatorTooltipEl.style.top = `${top}px`;
        indicatorTooltipEl.style.left = `${left}px`;
        indicatorTooltipEl.style.display = "block";
      }

      function scheduleIndicatorTooltip(rowEl, helpText) {
        clearIndicatorTooltipTimer();
        hideIndicatorTooltip();
        indicatorTooltipTimer = setTimeout(() => {
          showIndicatorTooltip(rowEl, helpText);
        }, 1000);
      }

      function hasValidSnapshotId(symbol, snapshotId) {
        if (!symbol || !snapshotId) return false;
        const items = savedSnapshotItemsBySymbol[symbol] || [];
        return items.some((x) => x && x.id === snapshotId);
      }

      function renderSavedSnapshotDropdown(symbol) {
        if (!savedSnapshotSelectEl) return;
        savedSnapshotSelectEl.innerHTML = "";
        const items = savedSnapshotItemsBySymbol[symbol] || [];
        if (!items.length) {
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "No saved snapshots";
          savedSnapshotSelectEl.appendChild(opt);
          savedSnapshotSelectEl.disabled = true;
          return;
        }

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Select snapshot...";
        savedSnapshotSelectEl.appendChild(placeholder);

        for (const item of items) {
          const opt = document.createElement("option");
          opt.value = String(item.id || "");
          opt.textContent = String(item.label || item.id || "snapshot");
          savedSnapshotSelectEl.appendChild(opt);
        }

        savedSnapshotSelectEl.disabled = false;
        const selectedId = selectedSnapshotIdBySymbol[symbol] || "";
        if (selectedId && hasValidSnapshotId(symbol, selectedId)) {
          savedSnapshotSelectEl.value = selectedId;
        } else {
          savedSnapshotSelectEl.value = "";
        }
      }

      function snapshotContextText(noteRecord, aiSummaryMarkdown) {
        const note = noteRecord && typeof noteRecord === "object" ? noteRecord : {};
        const title = String(note.title || "").trim();
        const createdAt = String(note.created_at || "").trim();
        const savedAt = String(note.saved_at || "").trim();
        const timeframe = String(note.timeframe || "").trim();
        const userBody = String(note.body || "").trim();
        const aiBody = String(aiSummaryMarkdown || "").trim();

        const lines = [];
        lines.push(`Title: ${title || "-"}`);
        lines.push(`Created (user): ${createdAt || "-"}`);
        lines.push(`Saved: ${savedAt || "-"}`);
        lines.push(`Timeframe: ${timeframe || "-"}`);
        lines.push("");
        lines.push("User Notes:");
        lines.push(userBody || "(no user notes)");
        lines.push("");
        lines.push("AI Notes:");
        lines.push(aiBody || "(no AI summary saved)");
        return lines.join("\\n");
      }

      function applySnapshotToChartState(symbol, noteRecord) {
        if (!currentRawPayload) return;
        if (!noteRecord || typeof noteRecord !== "object") return;
        const snap = noteRecord.snapshot && typeof noteRecord.snapshot === "object" ? noteRecord.snapshot : {};

        const snapTfNote = typeof noteRecord.timeframe === "string" ? noteRecord.timeframe : "";
        const snapTfState = typeof snap.timeframe === "string" ? snap.timeframe : "";
        const snapTf = snapTfNote || snapTfState || selectedTimeframeId;
        if (TIMEFRAME_DEFS.some((t) => t.id === snapTf)) {
          selectedTimeframeId = snapTf;
          if (timeframeSelectEl) timeframeSelectEl.value = snapTf;
        }

        const toggles = snap.toggles && typeof snap.toggles === "object" ? snap.toggles : {};
        for (const def of SERIES_DEFS) {
          if (typeof toggles[def.id] === "boolean") {
            visibilityState[def.id] = toggles[def.id];
          }
        }

        const renderModes = snap.series_render_modes && typeof snap.series_render_modes === "object"
          ? snap.series_render_modes
          : {};
        for (const def of SERIES_DEFS) {
          const mode = renderModes[def.id];
          if (typeof mode === "string") {
            seriesStyleState[def.id] = normalizeSeriesStyle(mode);
          }
        }

        if (snap.selected_metric_id === null) {
          selectedMetricScaleId = "";
        } else if (typeof snap.selected_metric_id === "string") {
          selectedMetricScaleId = snap.selected_metric_id;
        }

        const viewState = snap.view_state && typeof snap.view_state === "object" ? snap.view_state : {};
        const state = getViewState(symbol);
        state.xRange =
          Array.isArray(viewState.x_range) && viewState.x_range.length === 2
            ? [viewState.x_range[0], viewState.x_range[1]]
            : null;
        state.priceRange =
          Array.isArray(viewState.price_range) && viewState.price_range.length === 2
            ? [viewState.price_range[0], viewState.price_range[1]]
            : null;
        state.metricRanges = {};
        if (viewState.metric_ranges && typeof viewState.metric_ranges === "object") {
          for (const [k, v] of Object.entries(viewState.metric_ranges)) {
            if (Array.isArray(v) && v.length === 2) {
              state.metricRanges[k] = [v[0], v[1]];
            }
          }
        }

        currentPayload = transformPayloadForCurrentTimeframe(currentRawPayload);
        lastTogglePanelSignature = "";
        renderSeries(currentPayload, symbol);
      }

      async function loadSavedSnapshot(symbol, snapshotId) {
        if (!symbol || !snapshotId) return;
        try {
          const res = await fetch(
            `/api/notes/snapshot?symbol=${encodeURIComponent(symbol)}&snapshot_id=${encodeURIComponent(snapshotId)}`
          );
          const payload = await res.json();
          if (!res.ok || !payload?.ok) {
            const errMsg = payload?.error ? String(payload.error) : `HTTP ${res.status}`;
            setMetaMessage(`Snapshot load failed: ${errMsg}`);
            hideSavedSnapshotText();
            return;
          }

          const note = payload.note || {};
          const aiSummaryMarkdown = String(payload.ai_summary_markdown || "");
          setSavedSnapshotText(snapshotContextText(note, aiSummaryMarkdown), true);
          applySnapshotToChartState(symbol, note);
        } catch (err) {
          setMetaMessage(`Snapshot load failed: ${String(err)}`);
          hideSavedSnapshotText();
        }
      }

      async function refreshSavedSnapshotsForSymbol(symbol) {
        if (!symbol) return;
        hideSavedSnapshotText();
        try {
          const res = await fetch(`/api/notes/snapshots?symbol=${encodeURIComponent(symbol)}`);
          const payload = await res.json();
          if (currentSymbol() !== symbol) return;
          if (!res.ok) {
            const errMsg = payload?.error ? String(payload.error) : `HTTP ${res.status}`;
            setMetaMessage(`Failed to load saved snapshots for ${symbol}: ${errMsg}`);
            savedSnapshotItemsBySymbol[symbol] = [];
            renderSavedSnapshotDropdown(symbol);
            return;
          }
          const items = Array.isArray(payload?.snapshots) ? payload.snapshots : [];
          savedSnapshotItemsBySymbol[symbol] = items;
          renderSavedSnapshotDropdown(symbol);

          if (!items.length) {
            selectedSnapshotIdBySymbol[symbol] = "";
            hideSavedSnapshotText();
            return;
          }

          const selectedId = selectedSnapshotIdBySymbol[symbol] || "";
          if (selectedId && hasValidSnapshotId(symbol, selectedId)) {
            await loadSavedSnapshot(symbol, selectedId);
            return;
          }
          hideSavedSnapshotText();
        } catch (err) {
          if (currentSymbol() !== symbol) return;
          setMetaMessage(`Failed to load saved snapshots for ${symbol}: ${String(err)}`);
          savedSnapshotItemsBySymbol[symbol] = [];
          renderSavedSnapshotDropdown(symbol);
        }
      }

      function defaultSeriesStyle(def) {
        if (def.type === "candlestick") return "line";
        if (def.id === "mark_close" || def.id === "index_close") return "line";
        if (def.type === "markers") return "dots";
        return "both";
      }

      function normalizeSeriesStyle(style) {
        if (style === "line" || style === "dots" || style === "both") return style;
        return "both";
      }

      function nextSeriesStyle(style) {
        const current = normalizeSeriesStyle(style);
        if (current === "line") return "dots";
        if (current === "dots") return "both";
        return "line";
      }

      function seriesStyleBadge(style) {
        const current = normalizeSeriesStyle(style);
        if (current === "line") return "—";
        if (current === "dots") return "•";
        return "—•";
      }

      function traceModeForSeries(def) {
        const style = normalizeSeriesStyle(seriesStyleState[def.id] || defaultSeriesStyle(def));
        if (style === "line") return "lines";
        if (style === "dots") return "markers";
        return "lines+markers";
      }

      function hasAnyFinite(values) {
        if (!Array.isArray(values)) return false;
        return values.some((v) => typeof v === "number" && Number.isFinite(v));
      }

      function seriesValues(data, def) {
        if (def.id === "trade_candle") return data.kline_close || [];
        return data[def.key] || [];
      }

      function toLineTrace(x, y, name, color, axisRef, mode = "lines", lineWidth = 1.2) {
        return {
          type: "scatter",
          x,
          y,
          name,
          mode,
          line: { width: lineWidth, color },
          marker: { size: 6, color },
          yaxis: axisRef,
          hoverinfo: "skip"
        };
      }

      function toCandleTrace(x, open, high, low, close, axisRef, name = "Trade Candle") {
        return {
          type: "candlestick",
          x,
          open,
          high,
          low,
          close,
          name,
          yaxis: axisRef,
          increasing: {
            line: { color: "#22c55e", width: 1.1 },
            fillcolor: "rgba(34,197,94,0.22)"
          },
          decreasing: {
            line: { color: "#ef4444", width: 1.1 },
            fillcolor: "rgba(239,68,68,0.22)"
          },
          whiskerwidth: 0.4,
          opacity: 0.62,
          hoverinfo: "skip"
        };
      }

      function setNoteStatus(msg, isError = false) {
        if (!noteStatus) return;
        noteStatus.textContent = String(msg || "");
        noteStatus.style.color = isError ? "#fda4af" : "#94a3b8";
      }

      function setSaveAsIsVisible(visible) {
        if (!saveAsIsBtn) return;
        saveAsIsBtn.classList.toggle("hidden", !visible);
      }

      function refreshGeminiActionState() {
        if (generateGeminiBtn) {
          generateGeminiBtn.textContent = aiPreviewFailed ? "Retry Gemini Draft" : "Generate Gemini Draft";
        }
        setSaveAsIsVisible(aiPreviewFailed);
      }

      function isNoteModalOpen() {
        return Boolean(noteModalBackdrop && noteModalBackdrop.classList.contains("open"));
      }

      function openNoteModal() {
        const symbol = currentSymbol();
        if (!symbol) {
          setMetaMessage("Select a symbol before adding a note.");
          return;
        }
        const nowIso = new Date().toISOString();
        if (noteSymbolInput) noteSymbolInput.value = symbol;
        if (noteDateInput) noteDateInput.value = nowIso;
        if (noteTitleInput) noteTitleInput.value = `${symbol} ${selectedTimeframeId} note`;
        if (noteBodyInput) noteBodyInput.value = "";
        if (noteAiSummaryInput) noteAiSummaryInput.value = "";
        currentAiPreview = null;
        aiPreviewAttempted = false;
        aiPreviewFailed = false;
        refreshGeminiActionState();
        setNoteStatus("");
        if (noteModalBackdrop) {
          noteModalBackdrop.classList.add("open");
          noteModalBackdrop.setAttribute("aria-hidden", "false");
        }
        setTimeout(() => {
          if (noteTitleInput) noteTitleInput.focus();
        }, 0);
      }

      function closeNoteModal() {
        if (noteModalBackdrop) {
          noteModalBackdrop.classList.remove("open");
          noteModalBackdrop.setAttribute("aria-hidden", "true");
        }
      }

      function setNoteButtonsBusy(isBusy) {
        const busy = Boolean(isBusy);
        if (generateGeminiBtn) generateGeminiBtn.disabled = busy;
        if (saveAsIsBtn) saveAsIsBtn.disabled = busy;
        if (saveNoteBtn) saveNoteBtn.disabled = busy;
      }

      function currentNoteFormData() {
        const symbol = currentSymbol();
        if (!symbol) {
          return { error: "No symbol selected." };
        }
        const title = String(noteTitleInput?.value || "").trim();
        const body = String(noteBodyInput?.value || "").trim();
        const createdAt = String(noteDateInput?.value || new Date().toISOString());
        if (!title) {
          return { error: "Title is required.", focus: noteTitleInput };
        }
        if (!body) {
          return { error: "Thoughts are required.", focus: noteBodyInput };
        }
        return { symbol, title, body, createdAt };
      }

      function markAiPreviewStale() {
        const previewStatus = String(currentAiPreview?.status || "");
        if (previewStatus === "ok") {
          currentAiPreview = null;
          aiPreviewAttempted = false;
          aiPreviewFailed = false;
          if (noteAiSummaryInput) noteAiSummaryInput.value = "";
          refreshGeminiActionState();
          setNoteStatus("Content changed. Generate Gemini draft again before final save.");
          return;
        }
        if (aiPreviewFailed) {
          setNoteStatus("Gemini failed for this note. Retry Gemini Draft or use Save As-Is.");
        }
      }

      function currentVisibleSeriesState() {
        const data = currentPayload?.data || {};
        const out = [];
        for (const def of SERIES_DEFS) {
          const hasData = hasAnyFinite(seriesValues(data, def));
          out.push({
            id: def.id,
            label: def.label,
            group: def.group,
            has_data: hasData,
            enabled: Boolean(visibilityState[def.id]),
            visible: Boolean(hasData && visibilityState[def.id]),
            scale_key: def.group === "metric" ? metricScaleKey(def.id) : "price",
          });
        }
        return out;
      }

      function collectCurrentSnapshot(symbol) {
        const state = getViewState(symbol);
        const fullLayout = chartEl?._fullLayout;
        const selectedScaleKey = selectedMetricScaleId ? metricScaleKey(selectedMetricScaleId) : "";
        const snapshot = {
          captured_at: new Date().toISOString(),
          symbol,
          timeframe: selectedTimeframeId,
          series_render_modes: { ...seriesStyleState },
          selected_metric_id: selectedMetricScaleId || null,
          selected_metric_scale_key: selectedScaleKey || null,
          toggles: { ...visibilityState },
          visible_series: currentVisibleSeriesState(),
          view_state: {
            x_range: state.xRange ? [...state.xRange] : null,
            price_range: state.priceRange ? [...state.priceRange] : null,
            metric_ranges: JSON.parse(JSON.stringify(state.metricRanges || {})),
          },
          chart_layout: {
            width: fullLayout?.width ?? null,
            height: fullLayout?.height ?? null,
            xaxis_range: Array.isArray(fullLayout?.xaxis?.range) ? [...fullLayout.xaxis.range] : null,
            yaxis_range: Array.isArray(fullLayout?.yaxis?.range) ? [...fullLayout.yaxis.range] : null,
            yaxis2_range: Array.isArray(fullLayout?.yaxis2?.range) ? [...fullLayout.yaxis2.range] : null,
            yaxis2_autorange: fullLayout?.yaxis2?.autorange ?? null,
          },
          chart_viewport: {
            client_width: chartEl?.clientWidth ?? null,
            client_height: chartEl?.clientHeight ?? null,
          },
          url: window.location.href,
          meta_text: metaText ? metaText.textContent : "",
          rows: currentPayload?.rows ?? null,
        };
        return snapshot;
      }

      async function captureChartImageDataUrl() {
        try {
          const imgW = Math.max(900, chartEl?.clientWidth || 900);
          const imgH = Math.max(520, chartEl?.clientHeight || 520);
          return await Plotly.toImage(chartEl, {
            format: "png",
            width: imgW,
            height: imgH,
            scale: 1,
          });
        } catch (_err) {
          return null;
        }
      }

      function aiStatusTextFromPayload(aiSummary) {
        const aiStatus = String(aiSummary?.status || "unknown");
        const aiSummaryFile = aiSummary?.summary_file ? String(aiSummary.summary_file) : "";
        const aiError = aiSummary?.error ? String(aiSummary.error) : "";
        const aiSource = aiSummary?.source ? String(aiSummary.source) : "";
        let aiStatusText = "";
        if (aiStatus === "ok" || aiStatus === "saved") {
          aiStatusText = aiSummaryFile ? ` | Gemini: saved (${aiSummaryFile})` : " | Gemini: saved";
        } else if (aiStatus === "skipped") {
          aiStatusText = " | Gemini: skipped by user";
        } else if (aiStatus === "disabled") {
          aiStatusText = " | Gemini: disabled";
        } else if (aiStatus === "missing_api_key") {
          aiStatusText = " | Gemini: missing API key";
        } else if (aiStatus === "empty_response") {
          aiStatusText = " | Gemini: empty response";
        } else if (aiStatus === "error") {
          aiStatusText = aiError ? ` | Gemini error: ${aiError}` : " | Gemini: error";
        } else if (aiStatus) {
          aiStatusText = ` | Gemini: ${aiStatus}`;
        }
        if (aiSource) {
          aiStatusText += ` | source: ${aiSource}`;
        }
        return aiStatusText;
      }

      async function submitNoteSave({ symbol, title, body, createdAt, aiSummaryMarkdown, skipAiSummary, savingLabel }) {
        setNoteStatus(savingLabel || "Saving note...");
        setNoteButtonsBusy(true);
        const snapshot = collectCurrentSnapshot(symbol);
        const chartImageDataUrl = await captureChartImageDataUrl();

        try {
          const res = await fetch("/api/notes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              symbol,
              title,
              body,
              created_at: createdAt,
              timeframe: selectedTimeframeId,
              snapshot,
              chart_image_data_url: chartImageDataUrl,
              ai_summary_markdown: aiSummaryMarkdown || "",
              ai_preview: currentAiPreview,
              skip_ai_summary: Boolean(skipAiSummary),
            }),
          });
          const payload = await res.json();
          if (!res.ok || !payload?.ok) {
            const errMsg = payload?.error ? String(payload.error) : `HTTP ${res.status}`;
            setNoteStatus(`Save failed: ${errMsg}`, true);
            return;
          }
          const file = String(payload.note_file || "");
          const noteDir = String(payload.note_dir || "");
          const imgSaved = Boolean(payload.image_saved);
          const aiSummary = payload?.ai_summary || {};
          const aiStatusText = aiStatusTextFromPayload(aiSummary);
          const locationText = noteDir ? noteDir : file;
          setNoteStatus(`Saved: ${locationText}${imgSaved ? " (+ chart PNG)" : ""}${aiStatusText}`);
          setMetaMessage(`Note saved for ${symbol}: ${title}`);
          const savedSnapshotId = noteDir ? String(noteDir).split("/").filter(Boolean).pop() : "";
          if (savedSnapshotId) {
            selectedSnapshotIdBySymbol[symbol] = savedSnapshotId;
          }
          await refreshSavedSnapshotsForSymbol(symbol);
        } catch (err) {
          setNoteStatus(`Save failed: ${String(err)}`, true);
        } finally {
          setNoteButtonsBusy(false);
        }
      }

      async function generateGeminiDraft() {
        const form = currentNoteFormData();
        if (form.error) {
          setNoteStatus(String(form.error), true);
          if (form.focus) form.focus.focus();
          return;
        }
        const symbol = String(form.symbol);
        const title = String(form.title);
        const body = String(form.body);
        const createdAt = String(form.createdAt);
        const snapshot = collectCurrentSnapshot(symbol);

        aiPreviewAttempted = true;
        aiPreviewFailed = false;
        refreshGeminiActionState();
        setNoteStatus("Generating Gemini draft...");
        setNoteButtonsBusy(true);
        try {
          const res = await fetch("/api/notes/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              symbol,
              title,
              body,
              created_at: createdAt,
              timeframe: selectedTimeframeId,
              snapshot,
            }),
          });
          const payload = await res.json();
          if (!res.ok || !payload?.ok) {
            const errMsg = payload?.error ? String(payload.error) : `HTTP ${res.status}`;
            currentAiPreview = { status: "error", error: errMsg };
            aiPreviewFailed = true;
            refreshGeminiActionState();
            setNoteStatus(`Gemini preview failed: ${errMsg}. Retry Gemini Draft or use Save As-Is.`, true);
            return;
          }

          const aiPreview = payload?.ai_preview || {};
          currentAiPreview = aiPreview;
          const aiStatus = String(aiPreview.status || "unknown");
          const aiError = aiPreview.error ? String(aiPreview.error) : "";
          const summaryMarkdown = aiPreview.summary_markdown ? String(aiPreview.summary_markdown) : "";

          if (aiStatus === "ok" && summaryMarkdown) {
            aiPreviewFailed = false;
            refreshGeminiActionState();
            if (noteAiSummaryInput) noteAiSummaryInput.value = summaryMarkdown;
            setNoteStatus("Gemini draft ready. Review/edit it, then click Save Final Note.");
            return;
          }

          aiPreviewFailed = true;
          refreshGeminiActionState();
          if (aiStatus === "disabled") {
            setNoteStatus("Gemini is disabled in config. Retry after enabling, or Save As-Is.", true);
            return;
          }
          if (aiStatus === "missing_api_key") {
            setNoteStatus("Gemini key missing. Set GEMINI_API_KEY in .env, retry, or Save As-Is.", true);
            return;
          }
          if (aiStatus === "empty_response") {
            setNoteStatus("Gemini returned empty response. Retry Gemini Draft or Save As-Is.", true);
            return;
          }
          const suffix = aiError ? `: ${aiError}` : "";
          setNoteStatus(`Gemini preview failed (${aiStatus})${suffix}. Retry Gemini Draft or Save As-Is.`, true);
        } catch (err) {
          currentAiPreview = { status: "error", error: String(err) };
          aiPreviewFailed = true;
          refreshGeminiActionState();
          setNoteStatus(`Gemini preview failed: ${String(err)}. Retry Gemini Draft or Save As-Is.`, true);
        } finally {
          setNoteButtonsBusy(false);
        }
      }

      async function saveCurrentNote() {
        const form = currentNoteFormData();
        if (form.error) {
          setNoteStatus(String(form.error), true);
          if (form.focus) form.focus.focus();
          return;
        }
        const symbol = String(form.symbol);
        const title = String(form.title);
        const body = String(form.body);
        const createdAt = String(form.createdAt);
        const previewStatus = String(currentAiPreview?.status || "");
        if (previewStatus !== "ok") {
          if (aiPreviewFailed) {
            setNoteStatus("Gemini preview failed. Retry Gemini Draft or use Save As-Is.", true);
            if (generateGeminiBtn) generateGeminiBtn.focus();
            return;
          }
          setNoteStatus("Generate Gemini draft first, then review/edit it before final save.", true);
          if (generateGeminiBtn) generateGeminiBtn.focus();
          return;
        }

        const aiSummaryMarkdown = String(noteAiSummaryInput?.value || "").trim();
        if (!aiSummaryMarkdown) {
          setNoteStatus("Gemini draft is empty. Retry Gemini Draft or use Save As-Is.", true);
          if (generateGeminiBtn) generateGeminiBtn.focus();
          return;
        }

        await submitNoteSave({
          symbol,
          title,
          body,
          createdAt,
          aiSummaryMarkdown,
          skipAiSummary: false,
          savingLabel: "Saving final note...",
        });
      }

      async function saveNoteAsIs() {
        const form = currentNoteFormData();
        if (form.error) {
          setNoteStatus(String(form.error), true);
          if (form.focus) form.focus.focus();
          return;
        }
        if (!aiPreviewAttempted) {
          setNoteStatus("Generate Gemini draft first. If it fails, Save As-Is will be available.", true);
          if (generateGeminiBtn) generateGeminiBtn.focus();
          return;
        }
        if (!aiPreviewFailed) {
          setNoteStatus("Gemini draft is available. Use Save Final Note, or retry Gemini if needed.", true);
          if (saveNoteBtn) saveNoteBtn.focus();
          return;
        }

        const symbol = String(form.symbol);
        const title = String(form.title);
        const body = String(form.body);
        const createdAt = String(form.createdAt);
        await submitNoteSave({
          symbol,
          title,
          body,
          createdAt,
          aiSummaryMarkdown: "",
          skipAiSummary: true,
          savingLabel: "Saving note as-is (without Gemini summary)...",
        });
      }

      function axisKeyFromRef(axisRef) {
        if (axisRef === "y") return "yaxis";
        return `yaxis${axisRef.slice(1)}`;
      }

      function metricScaleKey(metricId) {
        if (metricId === "buy_ratio" || metricId === "sell_ratio") {
          return "buy_sell_ratio";
        }
        return metricId;
      }

      function metricScaleLabel(scaleKey, metricLabel) {
        if (scaleKey === "buy_sell_ratio") return "Buy/Sell Ratio";
        return metricLabel;
      }

      function timeframeMinutes(timeframeId) {
        const tf = TIMEFRAME_DEFS.find((x) => x.id === timeframeId);
        return tf ? tf.minutes : 1;
      }

      function currentViewStateKey(symbol) {
        return `${symbol}::${selectedTimeframeId}`;
      }

      function isFiniteNumber(value) {
        return typeof value === "number" && Number.isFinite(value);
      }

      function addValue(accum, value) {
        if (!isFiniteNumber(value)) return accum;
        return isFiniteNumber(accum) ? (accum + value) : value;
      }

      function updateMax(accum, value) {
        if (!isFiniteNumber(value)) return accum;
        return isFiniteNumber(accum) ? Math.max(accum, value) : value;
      }

      function updateMin(accum, value) {
        if (!isFiniteNumber(value)) return accum;
        return isFiniteNumber(accum) ? Math.min(accum, value) : value;
      }

      function aggregatePayload(rawPayload, timeframeId) {
        if (!rawPayload || !rawPayload.data) return rawPayload;
        if (timeframeId === "1m") return rawPayload;

        const bucketMs = timeframeMinutes(timeframeId) * 60 * 1000;
        if (!Number.isFinite(bucketMs) || bucketMs <= 60 * 1000) return rawPayload;

        const data = rawPayload.data || {};
        const ts = data.ts || [];
        const buckets = new Map();

        for (let i = 0; i < ts.length; i += 1) {
          const t = Date.parse(ts[i]);
          if (!Number.isFinite(t)) continue;
          const b = Math.floor(t / bucketMs) * bucketMs;

          let row = buckets.get(b);
          if (!row) {
            row = {
              ts_ms: b,
              kline_open: null,
              kline_high: null,
              kline_low: null,
              kline_close: null,
              mark_close: null,
              index_close: null,
              premium_close: null,
              kline_volume: null,
              kline_turnover: null,
              open_interest: null,
              buy_ratio: null,
              sell_ratio: null,
              long_short_ratio: null,
              basis_bps: null
            };
            buckets.set(b, row);
          }

          const o = data.kline_open?.[i];
          const h = data.kline_high?.[i];
          const l = data.kline_low?.[i];
          const c = data.kline_close?.[i];
          if (row.kline_open === null && isFiniteNumber(o)) row.kline_open = o;
          if (isFiniteNumber(h)) row.kline_high = updateMax(row.kline_high, h);
          if (isFiniteNumber(l)) row.kline_low = updateMin(row.kline_low, l);
          if (isFiniteNumber(c)) {
            row.kline_close = c;
            if (row.kline_open === null) row.kline_open = c;
            row.kline_high = updateMax(row.kline_high, c);
            row.kline_low = updateMin(row.kline_low, c);
          }

          const mc = data.mark_close?.[i];
          const ic = data.index_close?.[i];
          const pc = data.premium_close?.[i];
          const oi = data.open_interest?.[i];
          const br = data.buy_ratio?.[i];
          const sr = data.sell_ratio?.[i];
          const lsr = data.long_short_ratio?.[i];
          const bb = data.basis_bps?.[i];
          if (isFiniteNumber(mc)) row.mark_close = mc;
          if (isFiniteNumber(ic)) row.index_close = ic;
          if (isFiniteNumber(pc)) row.premium_close = pc;
          if (isFiniteNumber(oi)) row.open_interest = oi;
          if (isFiniteNumber(br)) row.buy_ratio = br;
          if (isFiniteNumber(sr)) row.sell_ratio = sr;
          if (isFiniteNumber(lsr)) row.long_short_ratio = lsr;
          if (isFiniteNumber(bb)) row.basis_bps = bb;

          const vol = data.kline_volume?.[i];
          const tov = data.kline_turnover?.[i];
          row.kline_volume = addValue(row.kline_volume, vol);
          row.kline_turnover = addValue(row.kline_turnover, tov);
        }

        const ordered = Array.from(buckets.values()).sort((a, b) => a.ts_ms - b.ts_ms);
        const out = {
          ts: [],
          kline_open: [],
          kline_high: [],
          kline_low: [],
          kline_close: [],
          mark_close: [],
          index_close: [],
          premium_close: [],
          kline_volume: [],
          kline_turnover: [],
          open_interest: [],
          buy_ratio: [],
          sell_ratio: [],
          long_short_ratio: [],
          basis_bps: [],
          funding_ts: [],
          funding_rate: []
        };

        for (const row of ordered) {
          out.ts.push(new Date(row.ts_ms).toISOString());
          out.kline_open.push(row.kline_open);
          out.kline_high.push(row.kline_high);
          out.kline_low.push(row.kline_low);
          out.kline_close.push(row.kline_close);
          out.mark_close.push(row.mark_close);
          out.index_close.push(row.index_close);
          out.premium_close.push(row.premium_close);
          out.kline_volume.push(row.kline_volume);
          out.kline_turnover.push(row.kline_turnover);
          out.open_interest.push(row.open_interest);
          out.buy_ratio.push(row.buy_ratio);
          out.sell_ratio.push(row.sell_ratio);
          out.long_short_ratio.push(row.long_short_ratio);
          out.basis_bps.push(row.basis_bps);
        }

        const fundingBuckets = new Map();
        const fundingTs = data.funding_ts || [];
        const fundingRate = data.funding_rate || [];
        for (let i = 0; i < fundingTs.length; i += 1) {
          const t = Date.parse(fundingTs[i]);
          if (!Number.isFinite(t)) continue;
          const rate = fundingRate[i];
          if (!isFiniteNumber(rate)) continue;
          const b = Math.floor(t / bucketMs) * bucketMs;
          fundingBuckets.set(b, rate);
        }
        const fundingOrdered = Array.from(fundingBuckets.entries()).sort((a, b) => a[0] - b[0]);
        for (const [t, rate] of fundingOrdered) {
          out.funding_ts.push(new Date(t).toISOString());
          out.funding_rate.push(rate);
        }

        return {
          symbol: rawPayload.symbol,
          rows: out.ts.length,
          data: out
        };
      }

      function transformPayloadForCurrentTimeframe(rawPayload) {
        return aggregatePayload(rawPayload, selectedTimeframeId);
      }

      function getViewState(symbol) {
        const key = currentViewStateKey(symbol);
        if (!viewStateBySymbol[key]) {
          viewStateBySymbol[key] = { xRange: null, priceRange: null, metricRanges: {} };
        }
        return viewStateBySymbol[key];
      }

      function applySavedAxisRange(layout, axisKey, rangeValue) {
        if (!Array.isArray(rangeValue) || rangeValue.length !== 2) return;
        if (!layout[axisKey]) return;
        layout[axisKey].range = [rangeValue[0], rangeValue[1]];
        layout[axisKey].autorange = false;
      }

      function extractRelayoutRange(eventData, axisId) {
        const rangeKey = `${axisId}.range`;
        if (Array.isArray(eventData[rangeKey]) && eventData[rangeKey].length === 2) {
          return [eventData[rangeKey][0], eventData[rangeKey][1]];
        }
        const r0 = eventData[`${axisId}.range[0]`];
        const r1 = eventData[`${axisId}.range[1]`];
        if (r0 !== undefined && r1 !== undefined) {
          return [r0, r1];
        }
        return null;
      }

      function togglePanelSignature(seriesMeta) {
        return seriesMeta
          .map((s) => `${s.id}:${s.hasData ? 1 : 0}:${visibilityState[s.id] ? 1 : 0}:${seriesStyleState[s.id] || ""}`)
          .join("|");
      }

      function bindRelayoutCapture() {
        if (relayoutCaptureBound) return;
        if (typeof chartEl.on !== "function") return;

        chartEl.on("plotly_relayout", (eventData) => {
          if (isProgrammaticLayoutChange) return;
          if (!eventData || !currentPayload) return;
          const symbol = currentSymbol();
          if (!symbol) return;
          const state = getViewState(symbol);

          if (eventData["xaxis.autorange"] === true) {
            state.xRange = null;
          } else {
            const xRange = extractRelayoutRange(eventData, "xaxis");
            if (xRange) state.xRange = xRange;
          }

          if (eventData["yaxis.autorange"] === true) {
            state.priceRange = null;
          } else {
            const priceRange = extractRelayoutRange(eventData, "yaxis");
            if (priceRange) state.priceRange = priceRange;
          }

          if (!selectedMetricScaleId) return;
          const selectedScaleKey = metricScaleKey(selectedMetricScaleId);
          if (eventData["yaxis2.autorange"] === true) {
            delete state.metricRanges[selectedScaleKey];
          } else {
            const metricRange = extractRelayoutRange(eventData, "yaxis2");
            if (metricRange) state.metricRanges[selectedScaleKey] = metricRange;
          }
        });

        relayoutCaptureBound = true;
      }

      function snapshotCurrentLayoutState(symbol, metricIdForY2 = selectedMetricScaleId) {
        if (!symbol) return;
        const fullLayout = chartEl?._fullLayout;
        if (!fullLayout) return;
        const state = getViewState(symbol);

        if (fullLayout.xaxis?.autorange === true) {
          state.xRange = null;
        } else if (Array.isArray(fullLayout.xaxis?.range) && fullLayout.xaxis.range.length === 2) {
          state.xRange = [fullLayout.xaxis.range[0], fullLayout.xaxis.range[1]];
        }

        if (fullLayout.yaxis?.autorange === true) {
          state.priceRange = null;
        } else if (Array.isArray(fullLayout.yaxis?.range) && fullLayout.yaxis.range.length === 2) {
          state.priceRange = [fullLayout.yaxis.range[0], fullLayout.yaxis.range[1]];
        }

        if (!metricIdForY2) return;
        const scaleKey = metricScaleKey(metricIdForY2);
        if (fullLayout.yaxis2?.autorange === true) {
          delete state.metricRanges[scaleKey];
        } else if (Array.isArray(fullLayout.yaxis2?.range) && fullLayout.yaxis2.range.length === 2) {
          state.metricRanges[scaleKey] = [fullLayout.yaxis2.range[0], fullLayout.yaxis2.range[1]];
        }
      }

      function renderTogglePanel(seriesMeta) {
        clearIndicatorTooltipTimer();
        hideIndicatorTooltip();
        legendListEl.innerHTML = "";
        const visibleSeries = seriesMeta.filter((s) => s.hasData);
        const hiddenCount = seriesMeta.length - visibleSeries.length;

        for (const item of visibleSeries) {
          const row = document.createElement("label");
          row.className = "toggle-row";
          if (item.help) {
            row.title = item.help;
          }
          row.addEventListener("mouseenter", () => {
            if (item.help) scheduleIndicatorTooltip(row, item.help);
          });
          row.addEventListener("mouseleave", () => {
            clearIndicatorTooltipTimer();
            hideIndicatorTooltip();
          });

          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = Boolean(visibilityState[item.id]);
          cb.addEventListener("change", () => {
            visibilityState[item.id] = cb.checked;
            const symbol = currentSymbol();
            if (symbol && currentPayload) {
              snapshotCurrentLayoutState(symbol, selectedMetricScaleId);
              renderSeries(currentPayload, symbol);
            }
          });

          const swatch = document.createElement("span");
          swatch.className = "swatch";
          if (item.id === "trade_candle") {
            swatch.style.background = "linear-gradient(135deg, rgba(34,197,94,0.90), rgba(239,68,68,0.85))";
          } else {
            swatch.style.background = item.color;
          }

          const txt = document.createElement("span");
          txt.className = "toggle-label";
          txt.textContent = item.label;
          if (item.help) {
            txt.title = item.help;
          }

          const styleBtn = document.createElement("button");
          styleBtn.type = "button";
          styleBtn.className = "series-style-btn";
          const isCandle = item.id === "trade_candle";
          styleBtn.textContent = isCandle ? "▥" : seriesStyleBadge(seriesStyleState[item.id]);
          styleBtn.title = isCandle
            ? "Candle style fixed"
            : "Series style: click to cycle line / dots / both";
          const enabled = Boolean(visibilityState[item.id]) && !isCandle;
          styleBtn.disabled = !enabled;
          styleBtn.addEventListener("click", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            if (styleBtn.disabled) return;
            seriesStyleState[item.id] = nextSeriesStyle(seriesStyleState[item.id]);
            const symbol = currentSymbol();
            if (symbol && currentPayload) {
              snapshotCurrentLayoutState(symbol, selectedMetricScaleId);
              renderSeries(currentPayload, symbol);
            }
          });

          row.appendChild(cb);
          row.appendChild(swatch);
          row.appendChild(txt);
          row.appendChild(styleBtn);
          legendListEl.appendChild(row);
        }

        if (hiddenCount > 0) {
          const note = document.createElement("div");
          note.className = "meta";
          note.style.marginTop = "10px";
          note.textContent = `${hiddenCount} series hidden (no data for this symbol).`;
          legendListEl.appendChild(note);
        }
      }

      function syncMetricScaleSelector(metricSeries) {
        const hadSelected = metricSeries.some((s) => s.id === selectedMetricScaleId);
        if (!hadSelected) {
          selectedMetricScaleId = metricSeries.length > 0 ? metricSeries[0].id : "";
        }

        metricAxisSelectEl.innerHTML = "";
        for (const s of metricSeries) {
          const opt = document.createElement("option");
          opt.value = s.id;
          opt.textContent = s.label;
          metricAxisSelectEl.appendChild(opt);
        }

        if (metricSeries.length === 0) {
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "No metric selected";
          metricAxisSelectEl.appendChild(opt);
          metricAxisSelectEl.value = "";
          metricAxisSelectEl.disabled = true;
        } else {
          metricAxisSelectEl.disabled = false;
          metricAxisSelectEl.value = selectedMetricScaleId;
        }
      }

      function renderSeries(payload, symbol) {
        const data = payload.data || {};
        const x = data.ts || [];
        const viewState = getViewState(symbol);

        const seriesMeta = SERIES_DEFS.map((def) => ({
          ...def,
          hasData: hasAnyFinite(seriesValues(data, def))
        }));

        const priceSeries = seriesMeta.filter((s) => s.group === "price" && s.hasData && visibilityState[s.id]);
        const metricSeries = seriesMeta.filter((s) => s.group === "metric" && s.hasData && visibilityState[s.id]);
        syncMetricScaleSelector(metricSeries);

        const traces = [];

        if (priceSeries.some((s) => s.id === "trade_candle")) {
          traces.push(
            toCandleTrace(
              x,
              data.kline_open || [],
              data.kline_high || [],
              data.kline_low || [],
              data.kline_close || [],
              "y",
              "Trade Candle"
            )
          );
        }

        for (const s of priceSeries) {
          if (s.id === "trade_candle") continue;
          const mode = traceModeForSeries(s);
          traces.push(
            toLineTrace(
              x,
              data[s.key] || [],
              s.label,
              s.color,
              "y",
              mode,
              s.lineWidth || 1.2
            )
          );
        }

        const axisDefs = {};
        const axisRefByScaleKey = {};
        const metricCount = metricSeries.length;
        const selectedMetric = metricSeries.find((s) => s.id === selectedMetricScaleId) || null;
        const selectedScaleKey = selectedMetric ? metricScaleKey(selectedMetric.id) : null;

        const metricGroups = [];
        for (const s of metricSeries) {
          const scaleKey = metricScaleKey(s.id);
          let group = metricGroups.find((g) => g.scaleKey === scaleKey);
          if (!group) {
            group = {
              scaleKey,
              label: metricScaleLabel(scaleKey, s.label),
              color: s.color
            };
            metricGroups.push(group);
          }
        }
        if (selectedMetric && selectedScaleKey) {
          const selectedGroup = metricGroups.find((g) => g.scaleKey === selectedScaleKey);
          if (selectedGroup) {
            selectedGroup.label = metricScaleLabel(selectedScaleKey, selectedMetric.label);
            selectedGroup.color = selectedMetric.color;
          }
        }

        let hiddenAxisIdx = 3;
        for (const group of metricGroups) {
          const isSelected = Boolean(selectedScaleKey && group.scaleKey === selectedScaleKey);
          const axisRef = isSelected ? "y2" : `y${hiddenAxisIdx++}`;
          axisRefByScaleKey[group.scaleKey] = axisRef;
          const axisKey = axisKeyFromRef(axisRef);

          axisDefs[axisKey] = {
            title: isSelected ? group.label : "",
            overlaying: "y",
            side: "right",
            anchor: "x",
            showgrid: false,
            zeroline: false,
            automargin: Boolean(isSelected),
            visible: Boolean(isSelected),
            showticklabels: Boolean(isSelected),
            tickfont: { color: group.color, size: 11 },
            titlefont: { color: group.color, size: 12 },
            ticks: isSelected ? "outside" : "",
            tickcolor: group.color,
            ticklen: isSelected ? 5 : 0,
            fixedrange: !isSelected
          };

          const savedRange = viewState.metricRanges[group.scaleKey];
          if (Array.isArray(savedRange) && savedRange.length === 2) {
            axisDefs[axisKey].range = [savedRange[0], savedRange[1]];
            axisDefs[axisKey].autorange = false;
          }
        }

        for (const s of metricSeries) {
          const scaleKey = metricScaleKey(s.id);
          const axisRef = axisRefByScaleKey[scaleKey] || "y2";
          const y = data[s.key] || [];
          const xVals = s.xKey ? (data[s.xKey] || []) : x;
          traces.push(
            toLineTrace(
              xVals,
              y,
              s.label,
              s.color,
              axisRef,
              traceModeForSeries(s),
              s.lineWidth || 1.2
            )
          );
        }

        const rightMargin = selectedScaleKey ? 130 : 95;
        const baseColumnWidth = Math.floor(
          chartEl.parentElement?.getBoundingClientRect().width ||
          chartEl.getBoundingClientRect().width ||
          1200
        );
        const containerWidth = Math.max(900, baseColumnWidth);
        const containerHeight = Math.max(
          520,
          Math.floor(chartEl.clientHeight || chartEl.getBoundingClientRect().height || (window.innerHeight - 170))
        );
        const layout = {
          hovermode: false,
          dragmode: "pan",
          uirevision: `${symbol}:${selectedTimeframeId}`,
          template: "plotly_dark",
          paper_bgcolor: "#020617",
          plot_bgcolor: "#020617",
          showlegend: false,
          autosize: false,
          width: containerWidth,
          height: containerHeight,
          margin: { l: 75, r: rightMargin, t: 32, b: 45 },
          xaxis: { title: "Time (UTC)", rangeslider: { visible: false }, gridcolor: "#1e293b" },
          yaxis: { title: "Price", gridcolor: "#1e293b" },
          ...axisDefs
        };

        applySavedAxisRange(layout, "xaxis", viewState.xRange);
        applySavedAxisRange(layout, "yaxis", viewState.priceRange);
        if (selectedScaleKey) {
          applySavedAxisRange(layout, "yaxis2", viewState.metricRanges[selectedScaleKey]);
        }

        isProgrammaticLayoutChange = true;
        const reactResult = Plotly.react(chartEl, traces, layout, {
          responsive: true,
          scrollZoom: true,
          displaylogo: false
        });
        if (reactResult && typeof reactResult.finally === "function") {
          reactResult.finally(() => {
            isProgrammaticLayoutChange = false;
          });
        } else {
          isProgrammaticLayoutChange = false;
        }
        bindRelayoutCapture();

        const panelSig = togglePanelSignature(seriesMeta);
        if (panelSig !== lastTogglePanelSignature) {
          renderTogglePanel(seriesMeta);
          lastTogglePanelSignature = panelSig;
        }
        const scaleLabel = (selectedMetric && selectedScaleKey)
          ? metricScaleLabel(selectedScaleKey, selectedMetric.label)
          : "None";
        metaText.textContent = `${symbol} | TF: ${selectedTimeframeId} | Rows: ${payload.rows || 0} | Active metrics: ${metricCount} | Right scale: ${scaleLabel}`;

        const url = new URL(window.location.href);
        url.searchParams.set("symbol", symbol);
        url.searchParams.set("tf", selectedTimeframeId);
        history.replaceState({}, "", url);
      }

      async function loadSeries(symbol) {
        let payload;
        try {
          const res = await fetch(`/api/series?symbol=${encodeURIComponent(symbol)}`);
          if (!res.ok) {
            setMetaMessage(`Failed to load series for ${symbol}: HTTP ${res.status}`);
            return;
          }
          payload = await res.json();
        } catch (err) {
          setMetaMessage(`Failed to load series: ${String(err)}`);
          return;
        }
        if (payload && payload.error) {
          setMetaMessage(`Series error for ${symbol}: ${String(payload.error)}`);
          return;
        }
        currentRawPayload = payload;
        currentPayload = transformPayloadForCurrentTimeframe(payload);
        lastTogglePanelSignature = "";
        renderSeries(currentPayload, symbol);
      }

      async function setSymbol(symbol) {
        const newIdx = symbols.indexOf(symbol);
        if (newIdx < 0) return;
        idx = newIdx;
        selectEl.value = symbol;
        setButtons();
        await loadSeries(symbol);
        await refreshSavedSnapshotsForSymbol(symbol);
      }

      async function stepSymbol(step) {
        if (symbols.length === 0) return;
        const next = idx + step;
        if (next < 0 || next >= symbols.length) return;
        await setSymbol(symbols[next]);
      }

      async function init() {
        try {
          const res = await fetch("/api/symbols");
          if (!res.ok) {
            setMetaMessage(`Failed to load symbols: HTTP ${res.status}`);
            return;
          }
          const payload = await res.json();
          symbols = Array.isArray(payload.symbols) ? payload.symbols : [];
        } catch (err) {
          setMetaMessage(`Failed to load symbols: ${String(err)}`);
          return;
        }
        timeframeSelectEl.innerHTML = "";
        for (const tf of TIMEFRAME_DEFS) {
          const opt = document.createElement("option");
          opt.value = tf.id;
          opt.textContent = tf.label;
          timeframeSelectEl.appendChild(opt);
        }

        for (const sym of symbols) {
          const opt = document.createElement("option");
          opt.value = sym;
          opt.textContent = sym;
          selectEl.appendChild(opt);
        }

        if (symbols.length === 0) {
          setMetaMessage("No symbols available. Build views first.");
          return;
        }

        const params = new URLSearchParams(window.location.search);
        const requested = (params.get("symbol") || "").toUpperCase();
        const requestedTf = (params.get("tf") || "1m").toLowerCase();
        if (TIMEFRAME_DEFS.some((t) => t.id === requestedTf)) {
          selectedTimeframeId = requestedTf;
        } else {
          selectedTimeframeId = "1m";
        }
        timeframeSelectEl.value = selectedTimeframeId;
        const first = symbols.includes(requested) ? requested : symbols[0];
        await setSymbol(first);
      }

      prevBtn.addEventListener("click", () => { void stepSymbol(-1); });
      nextBtn.addEventListener("click", () => { void stepSymbol(1); });
      selectEl.addEventListener("change", () => { void setSymbol(selectEl.value); });
      if (savedSnapshotSelectEl) {
        savedSnapshotSelectEl.addEventListener("change", () => {
          const symbol = currentSymbol();
          if (!symbol) return;
          const snapshotId = String(savedSnapshotSelectEl.value || "");
          selectedSnapshotIdBySymbol[symbol] = snapshotId;
          if (!snapshotId) {
            hideSavedSnapshotText();
            return;
          }
          void loadSavedSnapshot(symbol, snapshotId);
        });
      }
      if (addNoteBtn) {
        addNoteBtn.addEventListener("click", () => {
          openNoteModal();
        });
      }
      if (cancelNoteBtn) {
        cancelNoteBtn.addEventListener("click", () => {
          closeNoteModal();
        });
      }
      if (generateGeminiBtn) {
        generateGeminiBtn.addEventListener("click", () => {
          void generateGeminiDraft();
        });
      }
      if (saveAsIsBtn) {
        saveAsIsBtn.addEventListener("click", () => {
          void saveNoteAsIs();
        });
      }
      if (saveNoteBtn) {
        saveNoteBtn.addEventListener("click", () => {
          void saveCurrentNote();
        });
      }
      if (noteTitleInput) {
        noteTitleInput.addEventListener("input", () => {
          markAiPreviewStale();
        });
      }
      if (noteBodyInput) {
        noteBodyInput.addEventListener("input", () => {
          markAiPreviewStale();
        });
      }
      if (noteModalBackdrop) {
        noteModalBackdrop.addEventListener("click", (ev) => {
          if (ev.target === noteModalBackdrop) {
            closeNoteModal();
          }
        });
      }
      timeframeSelectEl.addEventListener("change", () => {
        const nextTf = timeframeSelectEl.value || "1m";
        if (!TIMEFRAME_DEFS.some((t) => t.id === nextTf)) return;
        const symbol = currentSymbol();
        if (symbol) {
          snapshotCurrentLayoutState(symbol, selectedMetricScaleId);
        }
        selectedTimeframeId = nextTf;
        if (symbol && currentRawPayload) {
          currentPayload = transformPayloadForCurrentTimeframe(currentRawPayload);
          renderSeries(currentPayload, symbol);
        }
      });
      metricAxisSelectEl.addEventListener("change", () => {
        const symbol = currentSymbol();
        if (symbol) {
          snapshotCurrentLayoutState(symbol, selectedMetricScaleId);
        }
        selectedMetricScaleId = metricAxisSelectEl.value || "";
        if (symbol && currentPayload) {
          renderSeries(currentPayload, symbol);
        }
      });

      window.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape" && isNoteModalOpen()) {
          ev.preventDefault();
          closeNoteModal();
          return;
        }
        if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
        if (ev.key === "ArrowLeft") {
          ev.preventDefault();
          void stepSymbol(-1);
        } else if (ev.key === "ArrowRight") {
          ev.preventDefault();
          void stepSymbol(1);
        }
      });

      window.addEventListener("resize", () => {
        try {
          const h = Math.max(
            520,
            Math.floor(chartEl.clientHeight || chartEl.getBoundingClientRect().height || (window.innerHeight - 170))
          );
          const w = Math.max(
            900,
            Math.floor(
              chartEl.parentElement?.getBoundingClientRect().width ||
              chartEl.getBoundingClientRect().width ||
              1200
            )
          );
          isProgrammaticLayoutChange = true;
          Plotly.relayout(chartEl, { height: h, width: w });
          Plotly.Plots.resize(chartEl);
          isProgrammaticLayoutChange = false;
        } catch (_err) {
          isProgrammaticLayoutChange = false;
          // No-op before first render.
        }
      });
      window.addEventListener("error", (ev) => {
        const msg = ev?.error ? String(ev.error) : String(ev?.message || "Unknown error");
        setMetaMessage(`Frontend error: ${msg}`);
      });
      window.addEventListener("unhandledrejection", (ev) => {
        const reason = ev?.reason ? String(ev.reason) : "Unknown promise rejection";
        setMetaMessage(`Frontend promise error: ${reason}`);
      });
      refreshGeminiActionState();
      hideSavedSnapshotText();
      void init();
    </script>
  </body>
</html>
        """

    return app


def require_view(db_path: Path, view_name: str) -> bool:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.views WHERE table_schema='main' AND table_name = ?",
            [view_name],
        ).fetchone()
        return bool(row and row[0])
    finally:
        conn.close()


def can_bind_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def resolve_available_port(host: str, preferred_port: int, search_window: int = 20) -> int:
    if can_bind_port(host, preferred_port):
        return preferred_port
    for offset in range(1, search_window + 1):
        candidate = preferred_port + offset
        if can_bind_port(host, candidate):
            print(f"[warn] Port {preferred_port} is busy. Falling back to {candidate}.")
            return candidate
    raise RuntimeError(
        f"No available port found in range {preferred_port}-{preferred_port + search_window} for host {host}."
    )


def main() -> int:
    args = parse_args()
    load_dotenv_file(Path(".env"))
    if args.config:
        load_dotenv_file(args.config.parent / ".env")
    config = load_config(args.config)

    db_path = args.db_path or path_from_config(config, "paths", "duckdb_path")
    host = args.host or str(config.get("browser", {}).get("host", "127.0.0.1"))
    preferred_port = args.port or int(config.get("browser", {}).get("port", 8051))
    port = resolve_available_port(host, preferred_port)
    start_ts_ms = config.get("plot", {}).get("start_ts_ms")
    end_ts_ms = config.get("plot", {}).get("end_ts_ms")
    notes_root_cfg = str(config.get("paths", {}).get("notes_root", "outputs/notes"))
    notes_root = Path(notes_root_cfg)
    gemini_cfg = gemini_notes_config_from_dict(config.get("ai"))

    if not db_path.exists():
        print(f"[error] DuckDB not found: {db_path}")
        print("[hint] run: python src/build_duckdb_views.py")
        return 1
    if not require_view(db_path, "symbol_timeseries_1m"):
        print("[error] required view missing: symbol_timeseries_1m")
        print("[hint] run: python src/build_duckdb_views.py")
        return 1

    app = create_app(db_path, start_ts_ms, end_ts_ms, notes_root, gemini_cfg)
    bind_url = f"http://{host}:{port}"
    local_url = f"http://127.0.0.1:{port}"
    print(f"[start] {bind_url}")
    if bind_url == local_url:
        print(f"[link]  {bind_url}")
    else:
        print(f"[link]  {local_url}")
    if gemini_cfg.enabled:
        print(
            f"[ai]    Gemini enabled | model={gemini_cfg.model} | key_env={gemini_cfg.api_key_env}"
        )
    else:
        print("[ai]    Gemini disabled in config.")
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
