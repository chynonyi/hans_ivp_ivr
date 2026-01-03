import asyncio
import yaml
import pandas as pd
from pathlib import Path
from functions import DataService, IVPIVRProjectService, VisualizationService

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# MANUAL SETTINGS (Uncomment to choose yourself, comment to use best configs)
# MANUAL_SETTINGS = {
#     "Symbol": "TSLA",
#     "IVP_from": 30,
#     "IVP_to": 60,
#     "Type": "Stock",  # or ETF or Stock
# }
MANUAL_SETTINGS = None # (Uncomment to use best configs)


async def main():
    print("Starting Backtest")
    data_service = DataService(CONFIG)

    heatmaps_dir = Path(CONFIG["output"]["heatmaps_dir"])
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(CONFIG["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = reports_dir.parent / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    # Directory where optimize.py saves raw results
    opt_results_dir = Path("outputs/optimization_results")
    
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
            trades_df = pf.trades.records_readable
            trades_df['Config'] = f"{ivp_from}-{ivp_to}"
            trades_df.to_csv(
                trades_dir / f"{symbol}_trades.csv", index=False
            )
        except Exception as e:
            print(f"Failed to export trades for {symbol}: {e}")

        # Save Portfolio Stats PDF
        try:
            stats_pdf_path = reports_dir / f"{symbol}_stats.pdf"
            VisualizationService.save_portfolio_stats_pdf(
                pf=pf, 
                output_path=str(stats_pdf_path)
            )
            print(f"Saved Stats PDF: {stats_pdf_path}")
        except Exception as e:
            print(f"Failed to save stats PDF: {e}")

        # Create Heatmap
        opt_results_file = opt_results_dir / f"{symbol}_opt_results.csv"
        
        if opt_results_file.exists():
            try:
                opt_results_df = pd.read_csv(opt_results_file)
                
                # Check for "Period" column
                if "Period" in opt_results_df.columns:
                    periods = opt_results_df["Period"].unique()
                    print(f"Found periods for {symbol}: {periods}")
                    
                    for period in periods:
                        if period == "Full":
                            continue

                        # Filter for specific period
                        period_df = opt_results_df[opt_results_df["Period"] == period]
                        
                        # Determine filename suffix
                        suffix = f"_{period}"
                        heatmap_pdf_path = heatmaps_dir / f"{symbol}_heatmap{suffix}.pdf"
                        
                        VisualizationService.create_heatmap(
                            results_df=period_df,
                            metric_name="Total Return [%]",
                            output_path=str(heatmap_pdf_path),
                        )
                        print(f"Saved Heatmap ({period}): {heatmap_pdf_path}")
                else:
                    # Backward compatibility
                    heatmap_pdf_path = heatmaps_dir / f"{symbol}_heatmap.pdf"
                    VisualizationService.create_heatmap(
                        results_df=opt_results_df,
                        metric_name="Sharpe Ratio",
                        output_path=str(heatmap_pdf_path),
                    )
                    print(f"Saved Heatmap: {heatmap_pdf_path}")
            except Exception as e:
                print(f"Failed to create heatmap: {e}")
        else:
            print(f"Heatmap skipped: No optimization data found at {opt_results_file}")

if __name__ == "__main__":
    asyncio.run(main())
