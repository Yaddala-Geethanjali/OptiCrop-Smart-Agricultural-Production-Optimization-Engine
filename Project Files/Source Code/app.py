from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        values = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        scaled_values = scaler.transform(values)

        result = model.predict(scaled_values)[0]

        confidence = max(model.predict_proba(scaled_values)[0]) * 100
        confidence = round(confidence, 2)

        return render_template(
            'result.html',
            prediction=result,
            confidence=confidence
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"
if __name__ == "__main__":
    app.run(debug=True)