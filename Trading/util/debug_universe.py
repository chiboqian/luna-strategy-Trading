
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_API_SECRET")
# Force paper url if not set
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE, attributes="options_enabled")
assets = client.get_all_assets(search_params) # Note: attributes param usage depends on SDK version

print(f"Total Assets (options_enabled): {len(assets)}")
if assets:
    print(f"First Asset: {assets[0]}")
    print(f"First Asset Attributes: {assets[0].attributes}")
