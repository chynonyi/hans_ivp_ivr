import asyncio
import sys
import yaml
import pandas as pd
from pathlib import Path
from functions import DataService, IVPIVRProjectService, VisualizationService

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# MANUAL SETTINGS (Uncomment to choose yourself, comment to use best configs)
MANUAL_SETTINGS = {
    "Symbol": "TSLA",
    "IVP_from": 30,
    "IVP_to": 60,
    "Type": "Stock",  # or ETF or Stock
}
#  MANUAL_SETTINGS = None # (Uncomment to use best configs)


async def main():
    print("Starting Backtest")
    data_service = DataService(CONFIG)

    reports_dir = Path(CONFIG["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = reports_dir.parent / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    lookback = CONFIG["strategy"]["ivp_ivr_lookback"]

    # 1. Determine list of configs to run
    run_list = []

    if MANUAL_SETTINGS:
        print(f"Using MANUAL settings for {MANUAL_SETTINGS['Symbol']}")
        run_list.append(MANUAL_SETTINGS)
    else:
        # Load from optimization results
        best_configs_file = Path(CONFIG["output"]["best_configs_file"])
        if not best_configs_file.exists():
            print("Best configs file not found. Please run optimize.py first.")
            return
        configs_df = pd.read_csv(best_configs_file)
        run_list = configs_df.to_dict("records")

    # 2. Run Backtests
    for row in run_list:
        symbol = row["Symbol"]
        ticker_type = row.get("Type", "Stock")
        ivp_from = row["IVP_from"]
        ivp_to = row["IVP_to"]

        print(f"Backtesting {symbol} [{ivp_from}-{ivp_to}]...")

        is_etf = ticker_type == "ETF"
        df = data_service.load_data(symbol, is_etf=is_etf)

        if df is None:
            print(f"Skipping {symbol}: Data not found (run download.py)")
            continue

        # Calculate Indicators (Warmup)
        df = IVPIVRProjectService.calculate_ivp_ivr(df, lookback)

        # Slice for Test Period
        start_key = "stocks" if not is_etf else "etfs"
        opt_start = CONFIG["periods"][start_key]["optimization_start"]

        df_sliced = df[df.index >= pd.Timestamp(opt_start)].copy()

        if df_sliced.empty:
            print(f"Skipping {symbol}: No data after {opt_start}")
            continue

        # Run Strategy
        pf = IVPIVRProjectService.backtest_ivp_ivr_strategy(
            symbol_df=df_sliced, entry_range=f"{ivp_from}-{ivp_to}", symbol=symbol
        )

        # Export Trades
        try:
            pf.trades.records_readable.to_csv(
                trades_dir / f"{symbol}_trades.csv", index=False
            )
        except:
            pass

        # Export PDF
        if "Total Return [%]" not in row:
            row["Total Return [%]"] = pf.total_return() * 100

        VisualizationService.create_comprehensive_pdf(
            symbol=symbol,
            best_config=row,
            best_pf=pf,
            all_results=[],
            output_path=str(reports_dir / f"{symbol}_report.pdf"),
        )

    print(f"Done. Reports saved to {reports_dir}")


if __name__ == "__main__":
    asyncio.run(main())
