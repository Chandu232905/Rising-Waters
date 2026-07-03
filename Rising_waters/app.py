from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/flood_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    annual_rainfall = float(request.form['annual_rainfall'])
    monsoon_rainfall = float(request.form['monsoon_rainfall'])
    cloud_visibility = float(request.form['cloud_visibility'])
    humidity = float(request.form['humidity'])
    river_level = float(request.form['river_level'])

    features = np.array([[annual_rainfall,
                          monsoon_rainfall,
                          cloud_visibility,
                          humidity,
                          river_level]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "⚠ Flood Likely"
    else:
        result = "✅ No Flood Risk"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)