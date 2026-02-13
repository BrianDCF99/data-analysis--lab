#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lab_config import load_config, path_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create interactive single-symbol chart from DuckDB views.")
    parser.add_argument(
        "symbol_positional",
        nargs="?",
        help="Optional symbol shortcut. Example: python src/plot_symbol_timeseries.py AZTECUSDT",
    )
    parser.add_argument("--symbol", default=None, help="Symbol to plot, e.g. AZTECUSDT")
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
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path (overrides config).",
    )
    parser.add_argument(
        "--start-ts-ms",
        type=int,
        default=None,
        help="Optional lower timestamp bound (milliseconds).",
    )
    parser.add_argument(
        "--end-ts-ms",
        type=int,
        default=None,
        help="Optional upper timestamp bound (milliseconds).",
    )
    return parser.parse_args()


def add_line(
    fig: go.Figure,
    df: pd.DataFrame,
    y_col: str,
    name: str,
    color: str,
    secondary_y: bool = False,
    visible: bool | str = True,
) -> None:
    if y_col not in df.columns:
        return
    series = df[y_col]
    if series.isna().all():
        return
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=series,
            name=name,
            mode="lines",
            line={"width": 1.6, "color": color},
            visible=visible,
        ),
        secondary_y=secondary_y,
    )


def add_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    y_col: str,
    name: str,
    color: str,
    secondary_y: bool = False,
    visible: bool | str = "legendonly",
) -> None:
    if y_col not in df.columns:
        return
    series = df[y_col]
    if series.isna().all():
        return
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=series,
            name=name,
            mode="markers",
            marker={"size": 7, "color": color},
            visible=visible,
        ),
        secondary_y=secondary_y,
    )


def build_query() -> str:
    return """
        SELECT
          symbol,
          ts_ms,
          ts,
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
    """


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    symbol_raw = args.symbol_positional or args.symbol or config.get("plot", {}).get("default_symbol")
    if not isinstance(symbol_raw, str) or not symbol_raw.strip():
        print("[error] missing symbol. Provide positional symbol, --symbol, or config plot.default_symbol.")
        return 1
    symbol = symbol_raw.upper().strip()
    db_path = args.db_path or path_from_config(config, "paths", "duckdb_path")
    plots_root = path_from_config(config, "paths", "plots_root")
    output_html = args.output_html or (plots_root / f"{symbol}_timeseries.html")
    start_ts_ms = args.start_ts_ms if args.start_ts_ms is not None else config.get("plot", {}).get("start_ts_ms")
    end_ts_ms = args.end_ts_ms if args.end_ts_ms is not None else config.get("plot", {}).get("end_ts_ms")

    if not db_path.exists():
        print(f"[error] DuckDB file not found: {db_path}")
        return 1

    conn = duckdb.connect(str(db_path), read_only=True)

    df = conn.execute(
        build_query(),
        [symbol, start_ts_ms, start_ts_ms, end_ts_ms, end_ts_ms],
    ).fetchdf()

    if df.empty:
        print(f"[error] no rows found for symbol={symbol}")
        return 1

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
    conn.close()

    # Force UTC plotting regardless of local machine timezone.
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    if not funding_df.empty:
        funding_df["ts"] = pd.to_datetime(funding_df["ts_ms"], unit="ms", utc=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price curves default visible.
    add_line(fig, df, "kline_close", "Trade Close", "#0ea5e9", secondary_y=False, visible=True)
    add_line(fig, df, "mark_close", "Mark Close", "#22c55e", secondary_y=False, visible=True)
    add_line(fig, df, "index_close", "Index Close", "#f59e0b", secondary_y=False, visible=True)
    add_line(fig, df, "premium_close", "Premium Close", "#ec4899", secondary_y=False, visible="legendonly")

    # Non-price metrics start hidden; click legend to toggle.
    add_line(fig, df, "basis_bps", "Basis (bps)", "#ef4444", secondary_y=True, visible="legendonly")
    add_line(fig, df, "kline_volume", "Volume", "#64748b", secondary_y=True, visible="legendonly")
    add_line(fig, df, "kline_turnover", "Turnover", "#334155", secondary_y=True, visible="legendonly")
    add_line(fig, df, "open_interest", "Open Interest", "#8b5cf6", secondary_y=True, visible="legendonly")
    add_line(fig, df, "buy_ratio", "Buy Ratio", "#16a34a", secondary_y=True, visible="legendonly")
    add_line(fig, df, "sell_ratio", "Sell Ratio", "#dc2626", secondary_y=True, visible="legendonly")
    add_line(fig, df, "long_short_ratio", "Long/Short Ratio", "#0f766e", secondary_y=True, visible="legendonly")

    if not funding_df.empty:
        add_markers(
            fig,
            funding_df.rename(columns={"funding_rate": "funding_rate_points"}),
            "funding_rate_points",
            "Funding Rate",
            "#111827",
            secondary_y=True,
            visible="legendonly",
        )

    fig.update_layout(
        title=f"{symbol} - Bybit Perp Time Series",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 60, "t": 80, "b": 40},
    )
    fig.update_xaxes(title_text="Time (UTC)")
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Alt Metrics", secondary_y=True)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    print(f"[done] wrote {output_html} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
