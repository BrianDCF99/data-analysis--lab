#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
from lab_config import load_config, path_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bybit perp symbol JSON/JSONL files into Parquet."
    )
    parser.add_argument(
        "symbol_positional",
        nargs="?",
        help="Optional single symbol shortcut. Example: python src/convert_symbol_to_parquet.py AZTECUSDT",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Config file path (default: config.json).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root folder containing symbol folders (overrides config).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for partitioned Parquet (overrides config).",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="Symbol to convert (repeatable). Example: --symbol AZTECUSDT --symbol BTCUSDT",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Text file with one symbol per line (overrides/extends config list).",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Convert every symbol folder under source-root.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Rows per write batch for JSONL conversions (overrides config).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Parquet compression codec (overrides config).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel symbol workers (overrides config).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet files.",
    )
    return parser.parse_args()


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return int(v)
        except ValueError:
            try:
                return int(float(v))
            except ValueError:
                return None
    return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            return None
    return None


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)$")


def parse_maybe_numeric(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        if _INT_RE.match(v):
            try:
                return int(v)
            except ValueError:
                return v
        if _FLOAT_RE.match(v):
            try:
                return float(v)
            except ValueError:
                return v
        return v
    return value


def ts_from_ms(ms: int | None) -> dt.datetime | None:
    if ms is None:
        return None
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc)


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            child = f"{prefix}_{key}" if prefix else str(key)
            flatten(child, inner, out)
        return
    if isinstance(value, list):
        out[prefix] = json.dumps(value, separators=(",", ":"))
        return
    out[prefix] = parse_maybe_numeric(value)


def write_parquet_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
    compression: str,
) -> int:
    if not rows:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression=compression)
    return table.num_rows


def output_is_fresh(output_path: Path, source_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        return output_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    except FileNotFoundError:
        return False


def convert_jsonl_file(
    input_path: Path,
    output_path: Path,
    transform: Callable[[dict[str, Any]], dict[str, Any] | None],
    chunk_size: int,
    compression: str,
    overwrite: bool,
) -> int:
    if output_path.exists():
        if not overwrite:
            if output_is_fresh(output_path, input_path):
                print(f"[skip] {output_path} already exists and is up-to-date")
                return 0
            print(f"[rebuild] {output_path} source changed")
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    row_count = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {input_path}:{line_no} JSON decode failed: {exc}")
                continue
            if not isinstance(payload, dict):
                continue
            row = transform(payload)
            if row is None:
                continue
            rows.append(row)

            if len(rows) >= chunk_size:
                table = pa.Table.from_pylist(rows) if writer is None else pa.Table.from_pylist(rows, schema=writer.schema)
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                writer.write_table(table)
                row_count += table.num_rows
                rows.clear()

    if rows:
        table = pa.Table.from_pylist(rows) if writer is None else pa.Table.from_pylist(rows, schema=writer.schema)
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
        writer.write_table(table)
        row_count += table.num_rows

    if writer is not None:
        writer.close()

    if row_count == 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        print(f"[warn] no rows in {input_path}")
    else:
        print(f"[ok] {input_path.name} -> {output_path.name} rows={row_count}")
    return row_count


def kline_transform(symbol: str) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    def _inner(payload: dict[str, Any]) -> dict[str, Any] | None:
        ts_ms = to_int(payload.get("ts"))
        raw = payload.get("raw")
        raw_list = raw if isinstance(raw, list) else []
        ts_from_raw_ms = to_int(raw_list[0]) if len(raw_list) >= 1 else None
        has_volume_turnover = len(raw_list) >= 7
        return {
            "symbol": symbol,
            "ts_ms": ts_ms,
            "ts": ts_from_ms(ts_ms),
            "ts_from_raw_ms": ts_from_raw_ms,
            "ts_from_raw": ts_from_ms(ts_from_raw_ms),
            "open": to_float(payload.get("open")),
            "high": to_float(payload.get("high")),
            "low": to_float(payload.get("low")),
            "close": to_float(payload.get("close")),
            "volume": to_float(payload.get("volume")),
            "turnover": to_float(payload.get("turnover")),
            "has_volume_turnover": has_volume_turnover,
            "raw_json": json.dumps(raw, separators=(",", ":")) if isinstance(raw, list) else None,
            "extra_json": json.dumps(payload.get("extra"), separators=(",", ":"))
            if isinstance(payload.get("extra"), list)
            else None,
        }

    return _inner


def funding_transform(default_symbol: str) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    def _inner(payload: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(payload.get("symbol") or default_symbol)
        collector_ts_ms = to_int(payload.get("ts"))
        funding_ts_ms = to_int(payload.get("fundingRateTimestamp"))
        ts_ms = collector_ts_ms if collector_ts_ms is not None else funding_ts_ms
        return {
            "symbol": symbol,
            "ts_ms": ts_ms,
            "ts": ts_from_ms(ts_ms),
            "collector_ts_ms": collector_ts_ms,
            "collector_ts": ts_from_ms(collector_ts_ms),
            "funding_rate_timestamp_ms": funding_ts_ms,
            "funding_rate_timestamp": ts_from_ms(funding_ts_ms),
            "funding_rate_timestamp_raw": payload.get("fundingRateTimestamp"),
            "funding_rate": to_float(payload.get("fundingRate")),
        }

    return _inner


def open_interest_transform(symbol: str) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    def _inner(payload: dict[str, Any]) -> dict[str, Any] | None:
        collector_ts_ms = to_int(payload.get("ts"))
        source_ts_ms = to_int(payload.get("timestamp"))
        ts_ms = collector_ts_ms if collector_ts_ms is not None else source_ts_ms
        return {
            "symbol": symbol,
            "ts_ms": ts_ms,
            "ts": ts_from_ms(ts_ms),
            "collector_ts_ms": collector_ts_ms,
            "collector_ts": ts_from_ms(collector_ts_ms),
            "source_timestamp_ms": source_ts_ms,
            "source_timestamp": ts_from_ms(source_ts_ms),
            "source_timestamp_raw": payload.get("timestamp"),
            "open_interest": to_float(payload.get("openInterest")),
        }

    return _inner


def account_ratio_transform(default_symbol: str) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    def _inner(payload: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(payload.get("symbol") or default_symbol)
        collector_ts_ms = to_int(payload.get("ts"))
        source_ts_ms = to_int(payload.get("timestamp"))
        ts_ms = collector_ts_ms if collector_ts_ms is not None else source_ts_ms
        buy = to_float(payload.get("buyRatio"))
        sell = to_float(payload.get("sellRatio"))
        long_short = None
        if buy is not None and sell not in (None, 0):
            long_short = buy / sell
        return {
            "symbol": symbol,
            "ts_ms": ts_ms,
            "ts": ts_from_ms(ts_ms),
            "collector_ts_ms": collector_ts_ms,
            "collector_ts": ts_from_ms(collector_ts_ms),
            "source_timestamp_ms": source_ts_ms,
            "source_timestamp": ts_from_ms(source_ts_ms),
            "source_timestamp_raw": payload.get("timestamp"),
            "buy_ratio": buy,
            "sell_ratio": sell,
            "long_short_ratio": long_short,
        }

    return _inner


def generic_jsonl_transform(symbol: str) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    def _inner(payload: dict[str, Any]) -> dict[str, Any] | None:
        row: dict[str, Any] = {"symbol": symbol}
        flatten("", payload, row)
        ts_ms = to_int(row.get("ts") or row.get("timestamp") or row.get("time"))
        row["ts_ms"] = ts_ms
        row["ts"] = ts_from_ms(ts_ms)
        return row

    return _inner


def convert_risk_limit(
    symbol_dir: Path,
    out_dir: Path,
    compression: str,
    overwrite: bool,
) -> int:
    risk_path = symbol_dir / "risk_limit.json"
    if not risk_path.exists():
        return 0
    out_path = out_dir / "risk_limit.parquet"
    if out_path.exists() and not overwrite:
        if output_is_fresh(out_path, risk_path):
            print(f"[skip] {out_path} already exists and is up-to-date")
            return 0
        print(f"[rebuild] {out_path} source changed")
        out_path.unlink()

    payload = json.loads(risk_path.read_text(encoding="utf-8"))
    symbol = str(payload.get("symbol") or symbol_dir.name)
    generated_iso = payload.get("generatedAtIso")
    rows = []
    for idx, entry in enumerate(payload.get("rows", []), start=1):
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "symbol": symbol,
                "generated_at_iso": generated_iso,
                "row_num": idx,
                "id": to_int(entry.get("id")),
                "risk_limit_value": to_float(entry.get("riskLimitValue")),
                "maintenance_margin": to_float(entry.get("maintenanceMargin")),
                "initial_margin": to_float(entry.get("initialMargin")),
                "is_lowest_risk": to_int(entry.get("isLowestRisk")),
                "max_leverage": to_float(entry.get("maxLeverage")),
                "mm_deduction": to_float(entry.get("mmDeduction")),
            }
        )
    count = write_parquet_rows(rows, out_path, compression=compression)
    if count > 0:
        print(f"[ok] risk_limit.json -> risk_limit.parquet rows={count}")
    else:
        print(f"[warn] no risk rows in {risk_path}")
    return count


def convert_flat_json_file(
    input_path: Path,
    out_path: Path,
    symbol: str,
    compression: str,
    overwrite: bool,
) -> int:
    if not input_path.exists():
        return 0
    if out_path.exists() and not overwrite:
        if output_is_fresh(out_path, input_path):
            print(f"[skip] {out_path} already exists and is up-to-date")
            return 0
        print(f"[rebuild] {out_path} source changed")
        out_path.unlink()

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(f"[warn] expected JSON object in {input_path}")
        return 0
    row: dict[str, Any] = {"symbol": symbol}
    flatten("", payload, row)
    count = write_parquet_rows([row], out_path, compression=compression)
    if count > 0:
        print(f"[ok] {input_path.name} -> {out_path.name} rows=1")
    return count


def convert_meta(
    symbol_dir: Path,
    out_dir: Path,
    compression: str,
    overwrite: bool,
) -> int:
    meta_path = symbol_dir / "meta.json"
    if not meta_path.exists():
        return 0
    out_path = out_dir / "meta.parquet"
    if out_path.exists() and not overwrite:
        if output_is_fresh(out_path, meta_path):
            print(f"[skip] {out_path} already exists and is up-to-date")
            return 0
        print(f"[rebuild] {out_path} source changed")
        out_path.unlink()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(f"[warn] expected JSON object in {meta_path}")
        return 0

    row: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "endpointState":
            row["endpoint_state_json"] = json.dumps(value, separators=(",", ":"))
            if isinstance(value, dict):
                for endpoint, endpoint_state in value.items():
                    if isinstance(endpoint_state, dict):
                        for k, v in endpoint_state.items():
                            row[f"{endpoint}_{k}"] = parse_maybe_numeric(v)
                    else:
                        row[endpoint] = parse_maybe_numeric(endpoint_state)
            continue
        row[key] = parse_maybe_numeric(value)

    symbol = str(row.get("symbol") or symbol_dir.name)
    row["symbol"] = symbol
    listing_ms = to_int(row.get("listingTimeMs"))
    collected_from_ms = to_int(row.get("collectedFromMs"))
    collected_to_ms = to_int(row.get("collectedToMs"))
    row["listing_time"] = ts_from_ms(listing_ms)
    row["collected_from"] = ts_from_ms(collected_from_ms)
    row["collected_to"] = ts_from_ms(collected_to_ms)

    count = write_parquet_rows([row], out_path, compression=compression)
    if count > 0:
        print(f"[ok] meta.json -> meta.parquet rows=1")
    return count


def convert_symbol(
    symbol: str,
    source_root: Path,
    output_root: Path,
    chunk_size: int,
    compression: str,
    overwrite: bool,
) -> tuple[int, int]:
    symbol_dir = source_root / symbol
    if not symbol_dir.exists() or not symbol_dir.is_dir():
        print(f"[warn] missing symbol folder: {symbol_dir}")
        return 0, 0

    out_dir = output_root / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    rows_processed = 0

    known_jsonl: set[Path] = set()

    for path in sorted(symbol_dir.glob("kline_*.jsonl")):
        known_jsonl.add(path)
        rows_processed += convert_jsonl_file(
            input_path=path,
            output_path=out_dir / f"{path.stem}.parquet",
            transform=kline_transform(symbol),
            chunk_size=chunk_size,
            compression=compression,
            overwrite=overwrite,
        )
        files_processed += 1

    for prefix in ("mark_price_kline_", "index_price_kline_", "premium_index_price_kline_"):
        for path in sorted(symbol_dir.glob(f"{prefix}*.jsonl")):
            known_jsonl.add(path)
            rows_processed += convert_jsonl_file(
                input_path=path,
                output_path=out_dir / f"{path.stem}.parquet",
                transform=kline_transform(symbol),
                chunk_size=chunk_size,
                compression=compression,
                overwrite=overwrite,
            )
            files_processed += 1

    funding_path = symbol_dir / "funding_history.jsonl"
    if funding_path.exists():
        known_jsonl.add(funding_path)
        rows_processed += convert_jsonl_file(
            input_path=funding_path,
            output_path=out_dir / "funding_history.parquet",
            transform=funding_transform(symbol),
            chunk_size=chunk_size,
            compression=compression,
            overwrite=overwrite,
        )
        files_processed += 1

    for path in sorted(symbol_dir.glob("open_interest_*.jsonl")):
        known_jsonl.add(path)
        rows_processed += convert_jsonl_file(
            input_path=path,
            output_path=out_dir / f"{path.stem}.parquet",
            transform=open_interest_transform(symbol),
            chunk_size=chunk_size,
            compression=compression,
            overwrite=overwrite,
        )
        files_processed += 1

    for path in sorted(symbol_dir.glob("account_ratio_*.jsonl")):
        known_jsonl.add(path)
        rows_processed += convert_jsonl_file(
            input_path=path,
            output_path=out_dir / f"{path.stem}.parquet",
            transform=account_ratio_transform(symbol),
            chunk_size=chunk_size,
            compression=compression,
            overwrite=overwrite,
        )
        files_processed += 1

    for path in sorted(symbol_dir.glob("*.jsonl")):
        if path in known_jsonl:
            continue
        rows_processed += convert_jsonl_file(
            input_path=path,
            output_path=out_dir / f"{path.stem}.parquet",
            transform=generic_jsonl_transform(symbol),
            chunk_size=chunk_size,
            compression=compression,
            overwrite=overwrite,
        )
        files_processed += 1

    risk_count = convert_risk_limit(symbol_dir, out_dir, compression=compression, overwrite=overwrite)
    if risk_count or (symbol_dir / "risk_limit.json").exists():
        files_processed += 1
        rows_processed += risk_count

    meta_count = convert_meta(symbol_dir, out_dir, compression=compression, overwrite=overwrite)
    if meta_count or (symbol_dir / "meta.json").exists():
        files_processed += 1
        rows_processed += meta_count

    raw_dir = symbol_dir / "raw"
    instrument_count = convert_flat_json_file(
        input_path=raw_dir / "instrument.json",
        out_path=out_dir / "instrument.parquet",
        symbol=symbol,
        compression=compression,
        overwrite=overwrite,
    )
    if instrument_count or (raw_dir / "instrument.json").exists():
        files_processed += 1
        rows_processed += instrument_count

    ticker_count = convert_flat_json_file(
        input_path=raw_dir / "ticker.json",
        out_path=out_dir / "ticker.parquet",
        symbol=symbol,
        compression=compression,
        overwrite=overwrite,
    )
    if ticker_count or (raw_dir / "ticker.json").exists():
        files_processed += 1
        rows_processed += ticker_count

    print(f"[symbol] {symbol} files={files_processed} rows={rows_processed}")
    return files_processed, rows_processed


def discover_symbols(source_root: Path) -> list[str]:
    symbols = []
    for child in sorted(source_root.iterdir()):
        if child.is_dir():
            symbols.append(child.name)
    return symbols


def read_symbols_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    symbols: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        symbols.append(s)
    return symbols


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    source_root: Path = args.source_root or path_from_config(config, "paths", "source_root")
    output_root: Path = args.output_root or path_from_config(config, "paths", "parquet_root")
    chunk_size = int(args.chunk_size if args.chunk_size is not None else config.get("convert", {}).get("chunk_size", 100000))
    compression = str(args.compression if args.compression is not None else config.get("convert", {}).get("compression", "zstd"))
    workers = int(args.workers if args.workers is not None else config.get("convert", {}).get("workers", 1))
    default_all_symbols = bool(config.get("convert", {}).get("default_all_symbols", True))
    overwrite = bool(args.overwrite or config.get("convert", {}).get("overwrite", False))
    config_symbols = config.get("convert", {}).get("symbols", [])
    if not isinstance(config_symbols, list):
        config_symbols = []
    config_symbols_file_raw = config.get("convert", {}).get("symbols_file")
    config_symbols_file = Path(config_symbols_file_raw) if isinstance(config_symbols_file_raw, str) and config_symbols_file_raw else None

    if not source_root.exists() or not source_root.is_dir():
        print(f"[error] source root does not exist or is not a folder: {source_root}")
        return 1

    symbols: list[str] = []
    if args.symbol_positional:
        symbols = [args.symbol_positional]
    if args.all_symbols:
        symbols = discover_symbols(source_root)
    symbols.extend(str(s) for s in config_symbols)
    if config_symbols_file is not None:
        symbols.extend(read_symbols_file(config_symbols_file))
    if args.symbols_file is not None:
        symbols.extend(read_symbols_file(args.symbols_file))
    if args.symbol:
        symbols.extend(args.symbol)
    if not symbols and default_all_symbols:
        symbols = discover_symbols(source_root)

    symbols = sorted(set(s.upper() for s in symbols if s.strip()))
    if not symbols:
        print("[error] no symbols selected.")
        return 1

    print(
        f"[start] symbols={len(symbols)} source={source_root} output={output_root} "
        f"chunkSize={chunk_size} compression={compression} overwrite={overwrite} workers={workers}"
    )

    total_files = 0
    total_rows = 0
    workers = max(1, int(workers))
    if workers == 1 or len(symbols) == 1:
        for symbol in symbols:
            files, rows = convert_symbol(
                symbol=symbol,
                source_root=source_root,
                output_root=output_root,
                chunk_size=chunk_size,
                compression=compression,
                overwrite=overwrite,
            )
            total_files += files
            total_rows += rows
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    convert_symbol,
                    symbol,
                    source_root,
                    output_root,
                    chunk_size,
                    compression,
                    overwrite,
                )
                for symbol in symbols
            ]
            for fut in as_completed(futures):
                files, rows = fut.result()
                total_files += files
                total_rows += rows

    print(f"[done] symbols={len(symbols)} files={total_files} rows={total_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
