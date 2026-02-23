from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your existing model (trained on all 13 features)
model = joblib.load('flood_model.pkl')
scaler = joblib.load('flood_scaler.pkl')
le_land = joblib.load('land_encoder.pkl')
le_soil = joblib.load('soil_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html',inputs=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.form.to_dict()
        
        # 1. Transform categorical inputs
        land_val = le_land.transform([data['land_cover']])[0]
        soil_val = le_soil.transform([data['soil_type']])[0]
        
        # 2. Neutral values for removed fields
        default_lat = 20.0
        default_long = 78.0
        default_pop_dense = 3000
        default_infra = 0

        # 3. Construct full 13-feature list
        full_features = [
            default_lat, default_long,
            float(data['rainfall']), float(data['temp']), 
            float(data['humidity']), float(data['discharge']),
            float(data['water_level']), float(data['elevation']), 
            land_val, soil_val,
            default_pop_dense, default_infra,
            int(data['history'])
        ]
        
        # 4. Predict
        scaled_features = scaler.transform([full_features])
        prediction = model.predict(scaled_features)
        
        result = "⚠️ DANGER: High Flood Risk!" if prediction[0] == 1 else "✅ SAFE: Low Flood Risk"
        bg = "danger" if prediction[0] == 1 else "success"

        # IMPORTANT: We pass 'inputs=data' back to the HTML
        return render_template('index.html', 
                               prediction_text=result, 
                               status=bg, 
                               inputs=data)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", status="warning")

if __name__ == '__main__':
    app.run(debug=True)