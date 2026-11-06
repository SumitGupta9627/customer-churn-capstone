"""Flask web app: prediction API + simple frontend.

Run from the project root:  python deployment/app.py
Then open http://127.0.0.1:5000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.config import CONTRACTS, PAYMENT_METHODS  # noqa: E402
from src.predict import load_model, predict_one  # noqa: E402

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
BUNDLE = load_model()


@app.get("/")
def home():
    return render_template("index.html", contracts=CONTRACTS, payments=PAYMENT_METHODS)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object"), 400
    try:
        return jsonify(predict_one(payload, BUNDLE))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
