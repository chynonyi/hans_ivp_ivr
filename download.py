import asyncio
import sys
import yaml
from functions import DataService

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


async def main():
    print("Starting Data Download")
    data_service = DataService(CONFIG)

    # Stocks
    start = CONFIG["periods"]["stocks"]["data_start"]
    end = CONFIG["periods"]["stocks"]["end"]
    stocks = CONFIG["tickers"]["stocks"]

    for symbol in stocks:
        await data_service.download_and_save_data(symbol, start, end, is_etf=False)

    # ETFs
    start_etf = CONFIG["periods"]["etfs"]["data_start"]
    end_etf = CONFIG["periods"]["etfs"]["end"]
    etfs = CONFIG["tickers"]["etfs"]  # dict mapping Stock -> ETF

    for base, etf_symbol in etfs.items():
        # For ETFs, we download the ETF data
        # Using base stock for IV (option_symbol=base)
        await data_service.download_and_save_data(
            etf_symbol, start_etf, end_etf, is_etf=True, option_symbol=base
        )

    print("--- Download Complete ---")


if __name__ == "__main__":
    asyncio.run(main())
