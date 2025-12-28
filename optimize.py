import asyncio
import sys
import yaml
import pandas as pd
from pathlib import Path
from functions import (
    DataService,
    IVPIVRProjectService,
    find_best_config,
    VisualizationService,
)

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


async def main():
    print("Starting Optimization")
    data_service = DataService(CONFIG)

    # Setup Output
    best_configs_file = Path(CONFIG["output"]["best_configs_file"])
    heatmaps_dir = Path(CONFIG["output"]["heatmaps_dir"])
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    best_configs_file.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # optimize settings
    lookback = CONFIG["strategy"]["ivp_ivr_lookback"]
    opt_start = CONFIG["strategy"]["optimization"]["start"]
    opt_end = CONFIG["strategy"]["optimization"]["end"]
    opt_step = CONFIG["strategy"]["optimization"]["step"]

    # 1. Stocks
    stocks = CONFIG["tickers"]["stocks"]
    opt_date_start = CONFIG["periods"]["stocks"]["optimization_start"]

    for symbol in stocks:
        print(f"Optimizing {symbol}...")
        df = data_service.load_data(symbol, is_etf=False)
        if df is None:
            print(f"Skipping {symbol} (No data - run download.py first)")
            continue

        opt_results = IVPIVRProjectService.run_optimization(
            symbol=symbol,
            df=df,
            ivp_ivr_lookback_days=lookback,
            optimization_step=opt_step,
            optimization_start=opt_start,
            optimization_end=opt_end,
            optimization_date_start=opt_date_start,
        )

        if not opt_results:
            continue

        best = find_best_config(opt_results)
        if best:
            best["Symbol"] = symbol
            best["Type"] = "Stock"
            results.append(best)

            # Heatmap
            heatmap_path = heatmaps_dir / f"{symbol}_optimization_heatmap.png"
            results_df = pd.DataFrame(opt_results)
            VisualizationService.create_heatmap(
                results_df=results_df,
                metric_name="Total Return [%]",
                output_path=str(heatmap_path),
            )

    # 2. ETFs
    etfs = CONFIG["tickers"]["etfs"]
    opt_date_start_etf = CONFIG["periods"]["etfs"]["optimization_start"]

    for base, etf_symbol in etfs.items():
        print(f"Optimizing {etf_symbol}...")
        df = data_service.load_data(etf_symbol, is_etf=True)
        if df is None:
            print(f"Skipping {etf_symbol} (No data - run download.py first)")
            continue

        opt_results = IVPIVRProjectService.run_optimization(
            symbol=etf_symbol,
            df=df,
            ivp_ivr_lookback_days=lookback,
            optimization_step=opt_step,
            optimization_start=opt_start,
            optimization_end=opt_end,
            optimization_date_start=opt_date_start_etf,
        )

        if not opt_results:
            continue

        best = find_best_config(opt_results)
        if best:
            best["Symbol"] = etf_symbol
            best["Type"] = "ETF"
            results.append(best)

            # Heatmap
            heatmap_path = heatmaps_dir / f"{etf_symbol}_optimization_heatmap.png"
            results_df = pd.DataFrame(opt_results)
            VisualizationService.create_heatmap(
                results_df=results_df,
                metric_name="Total Return [%]",
                output_path=str(heatmap_path),
            )

    # Save Results
    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values("Total Return [%]", ascending=False)
        df_res.to_csv(best_configs_file, index=False)
        print(f"Optimization results saved to {best_configs_file}")
    else:
        print("No results found.")


if __name__ == "__main__":
    asyncio.run(main())
