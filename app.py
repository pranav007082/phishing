from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from extract_features import extract_url_features

app = Flask(__name__)

# Load the pre-trained model and scaler (ensure these files are in your working directory)
model = load_model('221IT019_CNN_model.h5')
scaler = joblib.load('221IT019_scaler.pkl')

# Global variable to store prediction results
results_data = []  # Each entry: (URL, Prediction, Probability)
OUTPUT_CSV = "results.csv"

def update_output_csv():
    """Writes the current results_data to the OUTPUT_CSV file."""
    df = pd.DataFrame(results_data, columns=['URL', 'Prediction', 'Probability'])
    df.to_csv(OUTPUT_CSV, index=False)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({'error': 'Please enter a valid URL starting with http:// or https://'}), 400

    features = extract_url_features(url)
    print("Extracted features:", features)
    feature_names = list(features.keys())
    features_df = pd.DataFrame([features], columns=feature_names)

    try:
        features_scaled = scaler.transform(features_df)
        print("Scaled features:", features_scaled.flatten().tolist())
    except ValueError as e:
        print(f"Error scaling features: {e}")
        return jsonify({'error': 'Feature scaling mismatch with model expectations'}), 500

    num_features = features_df.shape[1]
    features_cnn = features_scaled.reshape((1, num_features, 1))
    prediction_prob = model.predict(features_cnn, verbose=0)[0][0]
    print("Prediction probability:", prediction_prob)
    result = 'phishing' if prediction_prob > 0.5 else 'benign'

    results_data.append((url, result, prediction_prob))
    update_output_csv()

    return jsonify({'prediction': result, 'probability': float(prediction_prob)})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        df = pd.read_csv(file)
        if 'URL' not in df.columns:
            return jsonify({'error': "CSV file must contain 'URL' column"}), 400

        batch_results = []
        for url in df['URL']:
            if not (isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))):
                batch_results.append((url, 'invalid URL', 0))
                continue
            features = extract_url_features(url)
            feature_names = list(features.keys())
            features_df = pd.DataFrame([features], columns=feature_names)
            try:
                features_scaled = scaler.transform(features_df)
            except ValueError:
                batch_results.append((url, 'scaling error', 0))
                continue

            num_features = features_df.shape[1]
            features_cnn = features_scaled.reshape((1, num_features, 1))
            prediction_prob = model.predict(features_cnn, verbose=0)[0][0]
            result = 'phishing' if prediction_prob > 0.5 else 'benign'
            batch_results.append((url, result, prediction_prob))

        results_data.extend(batch_results)
        update_output_csv()

        return jsonify({'processed': len(batch_results)})
    except Exception as e:
        print("Error during batch processing:", e)
        return jsonify({'error': 'Error processing CSV file'}), 500

@app.route('/download_output', methods=['GET'])
def download_output():
    if os.path.exists(OUTPUT_CSV):
        return send_file(OUTPUT_CSV, as_attachment=True)
    else:
        df = pd.DataFrame(columns=['URL', 'Prediction', 'Probability'])
        df.to_csv(OUTPUT_CSV, index=False)
        return send_file(OUTPUT_CSV, as_attachment=True)

if __name__ == '__main__':
    app.static_folder = '.'  # Serve static files (like index.html)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT",5000)))
