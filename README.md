# Data Analysis Lab

This folder is a scalable analytics workspace for Bybit perp datasets.
It is config-driven via `config.json`.

## Goals

- Convert raw JSON/JSONL files into columnar Parquet.
- Query large datasets with DuckDB (local OLAP engine).
- Generate interactive, toggleable time-series graphs per coin.
- Keep workflows scalable to 100GB+ datasets.

## Folder Layout

```text
data-analysis-lab/
  data/
    parquet/
      bybit_perp/
        symbol=AZTECUSDT/
          kline_1.parquet
          mark_price_kline_1.parquet
          ...
    duckdb/
      bybit_perp.duckdb
  outputs/
    plots/
      AZTECUSDT_timeseries.html
  src/
    lab_config.py
    convert_symbol_to_parquet.py
    build_duckdb_views.py
    plot_symbol_timeseries.py
    symbol_browser_app.py
  config.json
  requirements.txt
```

## Setup

```bash
cd /Users/briandelcarpio/Desktop/trading-platform/data-analysis-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Default Commands (Short)

Convert all symbols (uses `config.json` defaults):

```bash
python src/convert_symbol_to_parquet.py
```

Note:
- if `convert.default_all_symbols` is `false`, this command converts only symbols from `convert.symbols`/`convert.symbols_file`.
- if `convert.default_all_symbols` is `true`, this command converts every symbol directory.

Convert one symbol:

```bash
python src/convert_symbol_to_parquet.py AZTECUSDT
```

Build DuckDB views:

```bash
python src/build_duckdb_views.py
```

Plot one symbol:

```bash
python src/plot_symbol_timeseries.py AZTECUSDT
```

Plot default symbol from config:

```bash
python src/plot_symbol_timeseries.py
```

The output HTML graph supports toggling series on/off from the legend.

Run local browser app (dropdown + prev/next + keyboard arrows):

```bash
python src/symbol_browser_app.py
```

Then open:

```text
http://127.0.0.1:8051
```

### Notes + Gemini Structuring

When you save a note in the browser app:

- a dedicated note folder is created at `outputs/notes/<SYMBOL>/<TIMESTAMP>_<TITLE_SLUG>/`
- raw note is saved to `note.json`
- chart snapshot image is saved to `chart.png` (when available)
- Gemini prompt/response artifacts are saved as:
  - `gemini_prompt.txt`
  - `gemini_response.json`
  - `gemini_summary.md` (final edited summary)
- workflow in UI:
  - `Generate Gemini Draft` (gets model response)
  - edit the Gemini draft in the modal text box
  - `Save Final Note` (saves your edited version + artifacts)
  - if Gemini preview fails, use `Retry Gemini Draft` or `Save As-Is` (saves note without Gemini summary)
- left panel includes `Saved Snapshots` section:
  - dropdown lists saved snapshots for the currently selected coin
  - selecting one loads the saved chart state (toggles, timeframe, ranges)
  - below dropdown, a scrollable text area shows user notes + AI notes for that snapshot

Set your Gemini key before running:

```bash
export GEMINI_API_KEY="your_key_here"
```

Alternative:
- put `GEMINI_API_KEY=...` in `data-analysis-lab/.env` (auto-loaded by `symbol_browser_app.py`).

Gemini behavior is controlled in `config.json` under `ai`.

## Timestamp Notes

- `ts` in the JSONL files is normalized from Bybit source timestamps by the collector.
- We preserve separate timestamp fields in Parquet where applicable:
- funding: `collector_ts_ms`, `funding_rate_timestamp_ms`
- open interest/account ratio: `collector_ts_ms`, `source_timestamp_ms`
- meta timestamps are kept in `meta.parquet` (`listingTimeMs`, `collectedFromMs`, `collectedToMs` + converted datetime columns).
- No timestamp columns are merged away.

## Null Volume/Turnover Notes

- `mark-price-kline`, `index-price-kline`, and `premium-index-price-kline` often do not provide volume/turnover from Bybit.
- In those rows, `volume` and `turnover` being null is expected.
- We also preserve `raw_json` and `has_volume_turnover` so you can verify endpoint payload structure directly.

## What Gets Converted

- trade/mark/index/premium klines
- funding history
- open interest
- account ratio
- risk limits
- instrument snapshot
- ticker snapshot
- collector meta state

## Optional Overrides

You can still override config at runtime:

```bash
python src/convert_symbol_to_parquet.py AZTECUSDT --overwrite --workers 4
python src/convert_symbol_to_parquet.py --symbols-file config/symbols_user_batch.txt
python src/build_duckdb_views.py --db-path data/duckdb/custom.duckdb
python src/plot_symbol_timeseries.py AZTECUSDT --output-html outputs/plots/custom.html
python src/symbol_browser_app.py --port 8060
```

## Scale Notes

- Parquet is compressed, columnar, and efficient for large scans.
- DuckDB reads Parquet directly without loading everything into memory.
- Partitioning by `symbol=<TICKER>` allows symbol-level pruning.
- Conversion supports multi-processing via config `convert.workers`.

## When To Rebuild DuckDB Views

- Rebuild views after you add/update Parquet files (new conversion run).
- You do not need to rebuild repeatedly if Parquet has not changed.
- Fast rule: run `python src/build_duckdb_views.py` once after each conversion session.
