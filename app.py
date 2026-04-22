from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
model = joblib.load("model_v1.pkl")
@app.route("/")
def home():
 return "Model is running!"
@app.route("/predict", methods=["POST"])
def predict():
 data = request.json["features"]
 prediction = model.predict([data])
 return jsonify({"prediction": int(prediction[0])})
app.run(host="0.0.0.0", port=5000)