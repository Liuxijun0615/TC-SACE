# TC-SACE market data and evaluation inputs

Market data source: [BaoStock](https://www.baostock.com/). The research snapshot uses daily forward-adjusted OHLCV (`adjustflag=2`).

| Path | Contents |
| --- | --- |
| `ALL-stockdata/` | Original snapshot: 5,419 per-security CSVs (5,097 stock and 322 index files), with columns `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`. The study calendar runs from 2016-01-04 to 2025-12-31. |
| `data/folds/` | 13 frozen rolling panels for the study and `stage1_manifest.json` with fold dates, stock axes and SHA-256 digests. |
| `data/transition_bridge.json` | Quote and trade-status inputs for portfolio transitions between folds. |
| `scenario_evaluation_probes/` | 100 fixed evaluation portfolios per fold. |
| `normalization_reference_portfolios/` | 100 separate training reference portfolios per fold, with their own manifest. |

Each panel contains 817 dates by 100 symbols, with 61 warm-up, 504 training, 126 validation, and 126 test dates. The `phase` values 0–3 follow that order. Panel arrays include OHLCV, turnover amount, trading status, and ST status. Load NumPy `.npz` files with `numpy.load(path, allow_pickle=False)`. The fixed evaluation portfolios are distinct from the reference portfolios used for normalization.

Clone the repository or download its ZIP archive to obtain the complete CSV directory. These files are data inputs; the repository contains the core TC-SACE implementation rather than the complete paper experiment pipeline.
