import os
import asyncio
import pandas as pd
import yaml
import httpx
import ivolatility as ivol
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import vectorbt as vbt
import numpy as np

# Set matplotlib to use non-GUI backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load env
load_dotenv()

# API Keys
FMP_API_KEY = os.getenv("FMP_API_KEY")
IVOLATILITY_API_KEY = os.getenv("IVOLATILITY_API_KEY")

if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY not found in .env file")
if not IVOLATILITY_API_KEY:
    raise ValueError("IVOLATILITY_API_KEY not found in .env file")

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

STRATEGY_IV_MEAN_WINDOW = config["strategy"]["iv_mean_window"]
STRATEGY_IVP_IVR_LOOKBACK = config["strategy"]["ivp_ivr_lookback"]
BACKTEST_FREQ = config["backtest"]["freq"]
BACKTEST_INITIAL_CASH = config["backtest"]["initial_cash"]
BACKTEST_POSITION_SIZE = config["backtest"]["position_size"]
BACKTEST_SIZE_TYPE = config["backtest"]["size_type"]
BACKTEST_FEES = config["backtest"]["fees"]
BACKTEST_SLIPPAGE = config["backtest"]["slippage"]


class FMPService:
    # Shared HTTPX client with timeout
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=15.0)
            )
        return cls._client

    @classmethod
    async def fetch_with_retry(cls, url, retries=3, backoff=2):
        "Fetch data with retry logic"
        client = cls.get_client()
        for attempt in range(retries):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff**attempt)
                else:
                    raise e
            except Exception as e:
                raise e


class IvolatilityService:
    "IVolatility API Service"

    @staticmethod
    async def get_equity_ivx(
        symbol: str, region: str = "USA", from_: str = None, to: str = None
    ):
        "Get equity IVX data"
        ivol.setLoginParams(apiKey=IVOLATILITY_API_KEY)
        getMarketData = ivol.setMethod("/equities/eod/ivx")

        # Simple fetching
        marketData = getMarketData(symbol=symbol, from_=from_, to=to, region=region)
        return marketData.to_dict()


class DataService:
    def __init__(self, config):
        self.stocks_dir = Path(config["data"]["stocks_dir"])
        self.etfs_dir = Path(config["data"]["etfs_dir"])
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.etfs_dir.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, symbol: str, is_etf: bool = False) -> Path:
        directory = self.etfs_dir if is_etf else self.stocks_dir
        return directory / f"{symbol}.csv"

    async def download_and_save_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        is_etf: bool = False,
        option_symbol: str = None,
    ):
        """Downloads data from FMP and IVolatility and saves to CSV."""
        print(f"Downloading data for {symbol}...")

        # 1. Fetch Price Data
        url = (
            f"https://financialmodelingprep.com/stable/historical-price-eod/"
            f"dividend-adjusted/?symbol={symbol}&apikey={FMP_API_KEY}&from={start_date}&to={end_date}"
        )
        try:
            price_data = await FMPService.fetch_with_retry(url)
        except Exception as e:
            print(f"Failed to fetch price data for {symbol}: {e}")
            return False

        df = pd.DataFrame(price_data)
        if df.empty:
            print(f"No price data for {symbol}")
            return False

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df = df.set_index("date")

        # Rename cols to standard
        df = df.rename(
            columns={
                "adjOpen": "open",
                "adjClose": "close",
                "adjHigh": "high",
                "adjLow": "low",
                "volume": "volume",
            }
        )

        # 2. Fetch IV Data
        iv_symbol = option_symbol if option_symbol else symbol

        try:
            print(f"Fetching IV for {iv_symbol}...")
            iv_data = await IvolatilityService.get_equity_ivx(
                symbol=iv_symbol, from_=start_date, to=end_date
            )

            # Map IV to dates
            iv_map = {}
            if iv_data and "date" in iv_data:
                dates = iv_data["date"]  
                iv_vals = iv_data.get(f"{STRATEGY_IV_MEAN_WINDOW}d IV Mean", {})

                for idx, dt_str in dates.items():
                    val = iv_vals.get(idx)
                    iv_map[dt_str] = val

            # Add IV to dataframe
            # df.index is Timestamps
            df["iv"] = df.index.map(lambda x: iv_map.get(str(x.date())))

        except Exception as e:
            print(f"Error fetching IV for {symbol}: {e}")
            df["iv"] = None

        # Save to CSV
        file_path = self.get_file_path(symbol, is_etf)
        df.to_csv(file_path)
        print(f"Saved {symbol} to {file_path}")
        return True

    def load_data(self, symbol: str, is_etf: bool = False) -> pd.DataFrame:
        """Loads data from local CSV."""
        file_path = self.get_file_path(symbol, is_etf)
        if not file_path.exists():
            return None

        df = pd.read_csv(file_path, index_col="date", parse_dates=True)
        return df


def split_into_periods(
    start_date: str, end_date: str, window_years: int
) -> List[Tuple[str, str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    periods = []
    current = start

    while current < end:
        period_end = min(current + timedelta(days=365 * window_years), end)
        period_name = f"{current.year}-{period_end.year}"
        periods.append(
            (period_name, current.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"))
        )
        current = period_end

    return periods


class IVPIVRProjectService:
    "Main strategy service for IVP/IVR backtesting"

    @staticmethod
    def calculate_ivp_ivr(df: pd.DataFrame, lookback_days: int):
        # Calculate IVP/IVR on the dataframe
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")

        # Rolling window lookback
        roll = df["iv"].rolling(window=lookback_days, min_periods=lookback_days)

        # IV Rank
        df["ivr"] = (df["iv"] - roll.min()) / (roll.max() - roll.min()) * 100

        # IV Percentile
        def calc_percentile(x):
            if len(x) == 0:
                return np.nan
            current_iv = x.iloc[-1]
            return (x <= current_iv).mean() * 100

        df["ivp"] = roll.apply(calc_percentile, raw=False)
        df["ivp_ivr_blend"] = (df["ivp"] + df["ivr"]) / 2

        return df

    @staticmethod
    def optimize(
        symbol,
        symbol_df,
        optimization_step,
        optimization_start,
        optimization_end,
    ) -> list:
        results = []

        # Generate ranges: n1 is lower bound (X), n2 is upper bound (Y)
        n1_values = np.arange(optimization_start, optimization_end, optimization_step)

        for n1 in n1_values:
            n1 = round(n1, 3)
            # n2 must be >= n1
            n2_values = np.arange(
                n1 + optimization_step,
                optimization_end + optimization_step,
                optimization_step,
            )

            for n2 in n2_values:
                n2 = round(n2, 3)

                pf = IVPIVRProjectService.backtest_ivp_ivr_strategy(
                    symbol_df=symbol_df,
                    entry_range=f"{n1}-{n2}",
                    symbol=symbol,
                )

                pf_stats = pf.stats().to_dict()
                pf_stats["Total Return [%]"] = float(
                    pf_stats.get("Total Return [%]", 0) or 0
                )
                pf_stats["Open Trade PnL"] = float(
                    pf_stats.get("Open Trade PnL", 0) or 0
                )
                pf_stats["End Value"] = float(pf_stats.get("End Value", 1) or 1)

                pf_stats["Total Return [%]"] = (
                    0
                    if np.isnan(pf_stats["Total Return [%]"])
                    else pf_stats["Total Return [%]"]
                )
                pf_stats["Open Trade PnL"] = (
                    0
                    if np.isnan(pf_stats["Open Trade PnL"])
                    else pf_stats["Open Trade PnL"]
                )
                pf_stats["Total Return [%]"] = pf_stats["Total Return [%]"] + (
                    pf_stats["Open Trade PnL"] / pf_stats["End Value"] * 100
                )

                stats = {"IVP_from": n1, "IVP_to": n2}
                stats.update(pf_stats)

                results.append(stats)

        return results

    @staticmethod
    def run_optimization(
        symbol: str,
        df: pd.DataFrame,
        ivp_ivr_lookback_days: int,
        optimization_step: int,
        optimization_start: int,
        optimization_end: int,
        optimization_date_start: str = None,
    ):
        # 1. Calculate IVP/IVR on full data (including warmup)
        df = IVPIVRProjectService.calculate_ivp_ivr(df, ivp_ivr_lookback_days)

        # 2. Slice data for optimization period
        if optimization_date_start:
            df_opt = df[df.index >= pd.Timestamp(optimization_date_start)].copy()
        else:
            df_opt = df.copy()

        if df_opt.empty:
            print(f"Warning: No data for {symbol} after {optimization_date_start}")
            return []

        # 3. Optimize
        optimization_results_list = IVPIVRProjectService.optimize(
            symbol,
            df_opt,
            optimization_step,
            optimization_start,
            optimization_end,
        )

        return optimization_results_list

    @staticmethod
    def backtest_ivp_ivr_strategy(
        symbol_df: pd.DataFrame,
        entry_range: str,
        symbol: str,
    ):
        # Calculate positions
        # Assuming ivp_ivr_blend is already in symbol_df
        if "ivp_ivr_blend" not in symbol_df.columns:
            # Fallback if not present (should generally be present)
            pass

        symbol_df_positions = pd.Series(0, index=symbol_df.index)
        ivp_ivr_from = float(entry_range.split("-")[0])
        ivp_ivr_to = float(entry_range.split("-")[1])

        # Signal Generation
        symbol_df_positions_mask = (symbol_df["ivp_ivr_blend"] >= ivp_ivr_from) & (
            symbol_df["ivp_ivr_blend"] <= ivp_ivr_to
        )
        symbol_df_positions.loc[symbol_df_positions_mask] = 1

        pf = IVPIVRProjectService.backtest_with_vectorbt(
            symbol_df["close"], symbol_df_positions
        )

        return pf

    @staticmethod
    def backtest_with_vectorbt(price: pd.Series, positions: pd.Series):
        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=positions > 0,
            exits=positions == 0,
            size=positions * BACKTEST_POSITION_SIZE,
            size_type=BACKTEST_SIZE_TYPE,
            fees=BACKTEST_FEES,
            slippage=BACKTEST_SLIPPAGE,
            init_cash=BACKTEST_INITIAL_CASH,
            freq=BACKTEST_FREQ,
        )

        return pf


class VisualizationService:

    @staticmethod
    def save_portfolio_stats_pdf(pf, output_path: str):
        try:
            stats = pf.stats()
            stats_df = stats.to_frame()

            with PdfPages(output_path) as pdf:
                fig, ax = plt.subplots(figsize=(8.5, len(stats_df) * 0.4))
                ax.axis("off")
                table = ax.table(
                    cellText=stats_df.values,
                    colLabels=stats_df.columns,
                    rowLabels=stats_df.index,
                    cellLoc="center",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"Error saving portfolio stats PDF: {e}")
            raise e

    @staticmethod
    def create_heatmap(
        results_df: pd.DataFrame,
        metric_name: str,
        output_path: str,
        figsize: tuple = (16, 14),
    ):
        try:
            # Check if metric exists
            if metric_name not in results_df.columns:
                return False

            # Check required columns
            if (
                "IVP_from" not in results_df.columns
                or "IVP_to" not in results_df.columns
            ):
                return False

            # Create pivot table
            heatmap_data = results_df.pivot_table(
                index="IVP_from", columns="IVP_to", values=metric_name, aggfunc="mean"
            )

            print(f"Heatmap value range: min={heatmap_data.min().min()}, max={heatmap_data.max().max()}")

            # Check if heatmap has data
            if heatmap_data.empty:
                return False

            # Check for all NaN values
            if heatmap_data.isna().all().all():
                return False

            # Create heatmap
            plt.figure(figsize=figsize)

            # Determine colormap center based on metric
            center = 0 if "return" in metric_name.lower() else None

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn" if center == 0 else "viridis",
                center=center,
                cbar_kws={"label": metric_name},
                linewidths=0.5,
                linecolor="gray",
            )

            plt.title(f"{metric_name} Heatmap", fontsize=16, pad=20)
            plt.xlabel("IVP/IVR Upper Bound", fontsize=12)
            plt.ylabel("IVP/IVR Lower Bound", fontsize=12)
            plt.tight_layout()

            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return True

        except Exception as e:
            plt.close()
            print(f"Error creating heatmap: {e}")
            raise e


def find_best_config(results: List[Dict]) -> Dict:
    best_result = None
    best_return = -float("inf")

    for result in results:
        total_return = result.get("Total Return [%]", 0)
        if total_return > best_return:
            best_return = total_return
            best_result = result

    return best_result
