from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Create a Flask application
app = Flask(__name__)

# Load the scaler and models
scaler = joblib.load('scaler1.pkl')
models = {name: joblib.load(f'{name}_model.pkl') for name in ['RandomForest', 'SVM', 'KNN']}

# Function to convert numerical predictions to labels
def convert_prediction(pred):
    return 'healed' if pred == 1 else 'not healed'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the request
        data = request.form.to_dict()
        
        # Ensure all required features are in the request
        required_features = ['feature_0', 'feature_1', 'feature_2', 'texture_0', 'texture_1', 'texture_2', 
                             'texture_3', 'area', 'mean_intensity']
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing features in the request"}), 400
        
        # Extract features from the form data and scale them
        features = np.array([float(data[feature]) for feature in required_features]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Get predictions from each model and convert to labels
        predictions = {name: convert_prediction(int(model.predict(features_scaled)[0])) for name, model in models.items()}
        
        return render_template('results.html', predictions=predictions)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def entry():
    # Render the entry page
    return render_template('entry.html')

@app.route('/index', methods=['GET'])
def index():
    # Render the feature entry form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5008)
