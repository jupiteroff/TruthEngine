from flask import Flask, render_template, request, jsonify
import truthengine
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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
