from flask import Flask, render_template, request, jsonify
import truthengine
import traceback
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/live-prices')
def live_prices():
    """Get live prices for top cryptocurrencies"""
    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
        prices = {}
        
        # Add headers to avoid geo-blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        # Try multiple Binance endpoints
        base_urls = [
            "https://api.binance.us/api/v3",
            "https://api.binance.com/api/v3",
            "https://api1.binance.com/api/v3",
            "https://api2.binance.com/api/v3"
        ]
        
        for symbol in symbols:
            success = False
            for base_url in base_urls:
                try:
                    # Get current price
                    price_url = f"{base_url}/ticker/price?symbol={symbol}"
                    response = requests.get(price_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Get 24h price change
                    ticker_url = f"{base_url}/ticker/24hr?symbol={symbol}"
                    ticker_response = requests.get(ticker_url, headers=headers, timeout=5)
                    ticker_response.raise_for_status()
                    ticker_data = ticker_response.json()
                    
                    coin = symbol.replace('USDT', '')
                    prices[coin] = {
                        'price': float(data['price']),
                        'change_24h': float(ticker_data['priceChangePercent'])
                    }
                    success = True
                    break  # Success, move to next symbol
                except Exception as e:
                    continue  # Try next URL
            
            if not success:
                # If all endpoints failed, use a placeholder
                coin = symbol.replace('USDT', '')
                prices[coin] = {
                    'price': 0.0,
                    'change_24h': 0.0
                }
        
        return jsonify(prices)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        target = data.get('target')
        
        if not symbol or not target:
            return jsonify({"error": "Missing symbol or target"}), 400
            
        result = truthengine.get_prediction(symbol, target)
        
        # Convert datetime objects to string for JSON serialization
        result['target_dt'] = result['target_dt'].strftime('%Y-%m-%d %H:%M:%S %Z')
        result['current_time'] = result['current_time'].strftime('%Y-%m-%d %H:%M:%S %Z')
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
