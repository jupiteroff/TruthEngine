
import truthengine
import sys

print("Testing truthengine.get_prediction...")
try:
    # Test with a known valid symbol and target
    result = truthengine.get_prediction("BTC", "5m")
    print("Success!")
    print(f"Forecast: {result['forecast_price']}")
    print(f"PCT: {result['pct_change']}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
