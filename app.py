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
        
        for symbol in symbols:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Get 24h price change
            ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            ticker_response = requests.get(ticker_url, timeout=5)
            ticker_data = ticker_response.json()
            
            coin = symbol.replace('USDT', '')
            prices[coin] = {
                'price': float(data['price']),
                'change_24h': float(ticker_data['priceChangePercent'])
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
