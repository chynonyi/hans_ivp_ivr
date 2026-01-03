import asyncio
import yaml
import pandas as pd
from pathlib import Path
from functions import (
    DataService,
    IVPIVRProjectService,
    find_best_config,
    VisualizationService,
    split_into_periods,
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
    
    # New directory for raw optimization results
    opt_results_dir = Path("outputs/optimization_results")
    opt_results_dir.mkdir(parents=True, exist_ok=True)

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

        # Run FULL period optimization
        opt_results = IVPIVRProjectService.run_optimization(
            symbol=symbol,
            df=df,
            ivp_ivr_lookback_days=lookback,
            optimization_step=opt_step,
            optimization_start=opt_start,
            optimization_end=opt_end,
            optimization_date_start=opt_date_start,
        )
        
        # Tag with Period
        for res in opt_results:
            res["Period"] = "Full"

        print(f"Optimization for {symbol} (Full) was successful")

        # WINDOWED OPTIMIZATION START
        window_years = CONFIG["periods"]["stocks"].get("heatmap_window_years", 0)
        
        all_opt_results = []
        all_opt_results.extend(opt_results)
        
        if window_years > 0:
            print(f"Running windowed optimization for {symbol} ({window_years} years)...")
            
            # Determine end date
            end_date_str = CONFIG["periods"]["stocks"]["end"] 
            
            # Split into periods
            periods = split_into_periods(opt_date_start, end_date_str, window_years)
            
            for p_name, p_start, p_end in periods:
                print(f"  Optimizing {symbol} for period {p_name}...")
                
                p_start_dt = pd.Timestamp(p_start)
                p_end_dt = pd.Timestamp(p_end)
                
                # Slice DF to end at p_end
                df_period_end = df[df.index <= p_end_dt].copy()
                
                period_res = IVPIVRProjectService.run_optimization(
                    symbol=symbol,
                    df=df_period_end, # optimization will act on this
                    ivp_ivr_lookback_days=lookback,
                    optimization_step=opt_step,
                    optimization_start=opt_start,
                    optimization_end=opt_end,
                    optimization_date_start=p_start, # optimization will slice >= this
                )
                
                for r in period_res:
                    r["Period"] = p_name
                
                all_opt_results.extend(period_res)

        # WINDOWED OPTIMIZATION END

        if not opt_results:
             print(f"Optimization for {symbol} (Full) returned no results.")
             if not all_opt_results:
                 continue

        # Find best using ONLY "Full" results
        best = find_best_config(opt_results)
        if best:
            best["Symbol"] = symbol
            best["Type"] = "Stock"
            results.append(best)
            
            # Save ALL results (Full + Periods)
            results_df = pd.DataFrame(all_opt_results)
            results_df.to_csv(opt_results_dir / f"{symbol}_opt_results.csv", index=False)
    # 2. ETFs
    etfs = CONFIG["tickers"]["etfs"]
    opt_date_start_etf = CONFIG["periods"]["etfs"]["optimization_start"]

    for stock_symbol, etf_symbol in etfs.items():
        print(f"Optimizing {stock_symbol}:({etf_symbol})...")
        df = data_service.load_data(etf_symbol, is_etf=True)
        if df is None:
            print(f"Skipping {etf_symbol}: No data found for {etf_symbol}")
            continue

        # Run FULL period optimization
        opt_results = IVPIVRProjectService.run_optimization(
            symbol=etf_symbol,
            df=df,
            ivp_ivr_lookback_days=lookback,
            optimization_step=opt_step,
            optimization_start=opt_start,
            optimization_end=opt_end,
            optimization_date_start=opt_date_start_etf,
        )
        
        # Tag with Period
        for res in opt_results:
            res["Period"] = "Full"

        print(f"Optimization for {etf_symbol} (Full) was successful")

        # WINDOWED OPTIMIZATION START
        window_years = CONFIG["periods"]["etfs"].get("heatmap_window_years", 0)
        
        all_opt_results = []
        all_opt_results.extend(opt_results)
        
        if window_years > 0:
            print(f"Running windowed optimization for {etf_symbol} ({window_years} years)...")
            
            # Determine end date
            end_date_str = CONFIG["periods"]["etfs"]["end"] 
            
            # Split into periods
            periods = split_into_periods(opt_date_start_etf, end_date_str, window_years)
            
            for p_name, p_start, p_end in periods:
                print(f"  Optimizing {etf_symbol} for period {p_name}...")
                
                p_start_dt = pd.Timestamp(p_start)
                p_end_dt = pd.Timestamp(p_end)
                
                # Slice DF to end at p_end
                df_period_end = df[df.index <= p_end_dt].copy()
                
                period_res = IVPIVRProjectService.run_optimization(
                    symbol=etf_symbol,
                    df=df_period_end,
                    ivp_ivr_lookback_days=lookback,
                    optimization_step=opt_step,
                    optimization_start=opt_start,
                    optimization_end=opt_end,
                    optimization_date_start=p_start,
                )
                
                for r in period_res:
                    r["Period"] = p_name
                
                all_opt_results.extend(period_res)
        # WINDOWED OPTIMIZATION END

        if not opt_results:
            print(f"Optimization for {etf_symbol} (Full) returned no results.")
            if not all_opt_results:
                continue

        best = find_best_config(opt_results)
        if best:
            best["Symbol"] = etf_symbol
            best["Type"] = "ETF"
            results.append(best)

            # Save ALL results
            results_df = pd.DataFrame(all_opt_results)
            results_df.to_csv(opt_results_dir / f"{etf_symbol}_opt_results.csv", index=False)
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
