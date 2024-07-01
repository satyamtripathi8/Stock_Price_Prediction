from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

# Load the trained model
model = load('random_forest_model.joblib')

# Initialize Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        data = request.get_json()
        
        company = data['company']
        values = list(map(float, data['values']))

        # Prepare features for prediction
        features = np.array([values])

        # Perform prediction
        prediction = model.predict(features)

        # Prepare response as JSON
        response = {'prediction': prediction.tolist()}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
