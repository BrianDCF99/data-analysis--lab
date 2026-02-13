#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import duckdb
from lab_config import load_config, path_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DuckDB views over Bybit perp parquet partitions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Config file path (default: config.json).",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=None,
        help="Parquet root folder containing symbol=... partitions (overrides config).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="DuckDB file path (overrides config).",
    )
    return parser.parse_args()


def sql_quote(value: str) -> str:
    return value.replace("'", "''")


def glob_has_files(pattern: str) -> bool:
    return len(glob.glob(pattern)) > 0


def create_view_for_pattern(
    conn: duckdb.DuckDBPyConnection,
    parquet_root: Path,
    view_name: str,
    parquet_glob: str,
    extra_select: str = "",
) -> bool:
    pattern = str(parquet_root / parquet_glob)
    if not glob_has_files(pattern):
        print(f"[skip] {view_name}: no files for {pattern}")
        return False
    quoted = sql_quote(pattern)
    select_extra = f", {extra_select}" if extra_select else ""
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
          *,
          COALESCE(symbol, regexp_extract(filename, 'symbol=([^/]+)', 1)) AS symbol_partition
          {select_extra}
        FROM read_parquet('{quoted}', union_by_name=true, filename=true);
        """
    )
    count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
    print(f"[ok] {view_name}: rows={count}")
    return True


def view_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.views WHERE table_schema='main' AND table_name = ?",
        [name],
    ).fetchone()
    return bool(row and row[0])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    parquet_root: Path = args.parquet_root or path_from_config(config, "paths", "parquet_root")
    db_path: Path = args.db_path or path_from_config(config, "paths", "duckdb_path")

    if not parquet_root.exists():
        print(f"[error] parquet root not found: {parquet_root}")
        return 1

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    create_view_for_pattern(
        conn,
        parquet_root,
        "market_kline",
        "symbol=*/kline_*.parquet",
        "regexp_extract(filename, 'kline_([^/]+)\\.parquet$', 1) AS interval",
    )
    create_view_for_pattern(
        conn,
        parquet_root,
        "mark_price_kline",
        "symbol=*/mark_price_kline_*.parquet",
        "regexp_extract(filename, 'mark_price_kline_([^/]+)\\.parquet$', 1) AS interval",
    )
    create_view_for_pattern(
        conn,
        parquet_root,
        "index_price_kline",
        "symbol=*/index_price_kline_*.parquet",
        "regexp_extract(filename, 'index_price_kline_([^/]+)\\.parquet$', 1) AS interval",
    )
    create_view_for_pattern(
        conn,
        parquet_root,
        "premium_index_price_kline",
        "symbol=*/premium_index_price_kline_*.parquet",
        "regexp_extract(filename, 'premium_index_price_kline_([^/]+)\\.parquet$', 1) AS interval",
    )
    create_view_for_pattern(
        conn,
        parquet_root,
        "funding_history",
        "symbol=*/funding_history.parquet",
    )
    create_view_for_pattern(
        conn,
        parquet_root,
        "open_interest",
        "symbol=*/open_interest_*.parquet",
        "regexp_extract(filename, 'open_interest_([^/]+)\\.parquet$', 1) AS period",
    )
    create_view_for_pattern(
        conn,
        parquet_root,
        "account_ratio",
        "symbol=*/account_ratio_*.parquet",
        "regexp_extract(filename, 'account_ratio_([^/]+)\\.parquet$', 1) AS period",
    )
    create_view_for_pattern(conn, parquet_root, "risk_limit", "symbol=*/risk_limit.parquet")
    create_view_for_pattern(conn, parquet_root, "instrument", "symbol=*/instrument.parquet")
    create_view_for_pattern(conn, parquet_root, "ticker", "symbol=*/ticker.parquet")
    create_view_for_pattern(conn, parquet_root, "meta", "symbol=*/meta.parquet")

    required = [
        "market_kline",
        "mark_price_kline",
        "index_price_kline",
        "premium_index_price_kline",
        "open_interest",
        "account_ratio",
    ]
    if all(view_exists(conn, name) for name in required):
        conn.execute(
            """
            CREATE OR REPLACE VIEW symbol_timeseries_1m AS
            WITH k AS (
              SELECT
                COALESCE(symbol, symbol_partition) AS symbol,
                ts_ms,
                to_timestamp(ts_ms / 1000.0) AS ts,
                open AS kline_open,
                high AS kline_high,
                low AS kline_low,
                close AS kline_close,
                volume AS kline_volume,
                turnover AS kline_turnover
              FROM market_kline
              WHERE interval = '1'
            ),
            m AS (
              SELECT
                COALESCE(symbol, symbol_partition) AS symbol,
                ts_ms,
                open AS mark_open,
                high AS mark_high,
                low AS mark_low,
                close AS mark_close
              FROM mark_price_kline
              WHERE interval = '1'
            ),
            i AS (
              SELECT
                COALESCE(symbol, symbol_partition) AS symbol,
                ts_ms,
                open AS index_open,
                high AS index_high,
                low AS index_low,
                close AS index_close
              FROM index_price_kline
              WHERE interval = '1'
            ),
            p AS (
              SELECT
                COALESCE(symbol, symbol_partition) AS symbol,
                ts_ms,
                open AS premium_open,
                high AS premium_high,
                low AS premium_low,
                close AS premium_close
              FROM premium_index_price_kline
              WHERE interval = '1'
            ),
            oi AS (
              SELECT
                COALESCE(symbol, symbol_partition) AS symbol,
                ts_ms,
                open_interest
              FROM open_interest
              WHERE period = '5min' OR period IS NULL
            ),
            ar AS (
              SELECT
                COALESCE(symbol, symbol_partition) AS symbol,
                ts_ms,
                buy_ratio,
                sell_ratio,
                long_short_ratio
              FROM account_ratio
              WHERE period = '5min' OR period IS NULL
            )
            SELECT
              k.symbol,
              k.ts_ms,
              k.ts,
              k.kline_open,
              k.kline_high,
              k.kline_low,
              k.kline_close,
              k.kline_volume,
              k.kline_turnover,
              m.mark_open,
              m.mark_high,
              m.mark_low,
              m.mark_close,
              i.index_open,
              i.index_high,
              i.index_low,
              i.index_close,
              p.premium_open,
              p.premium_high,
              p.premium_low,
              p.premium_close,
              oi.open_interest,
              ar.buy_ratio,
              ar.sell_ratio,
              ar.long_short_ratio,
              CASE
                WHEN i.index_close IS NULL OR i.index_close = 0 OR m.mark_close IS NULL THEN NULL
                ELSE ((m.mark_close / i.index_close) - 1) * 10000
              END AS basis_bps
            FROM k
            LEFT JOIN m USING(symbol, ts_ms)
            LEFT JOIN i USING(symbol, ts_ms)
            LEFT JOIN p USING(symbol, ts_ms)
            LEFT JOIN oi USING(symbol, ts_ms)
            LEFT JOIN ar USING(symbol, ts_ms)
            ORDER BY symbol, ts_ms;
            """
        )

        timeseries_count = conn.execute("SELECT COUNT(*) FROM symbol_timeseries_1m").fetchone()[0]
        symbol_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM symbol_timeseries_1m").fetchone()[0]
        print(f"[ok] symbol_timeseries_1m: rows={timeseries_count} symbols={symbol_count}")
    else:
        print("[warn] symbol_timeseries_1m not created (missing one or more required views)")

    conn.close()
    print(f"[done] DuckDB ready: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
