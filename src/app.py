from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features).tolist()
    return jsonify({"prediction": prediction})

# ✅ This part is critical!
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
