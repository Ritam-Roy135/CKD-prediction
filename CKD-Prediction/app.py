from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ckd_prediction_pipeline.pkl")

model_bundle = joblib.load(MODEL_PATH)
models = model_bundle["models"]
default_model = model_bundle["default_model"]

NUMERIC_FEATURES = [
    "age", "bp", "sg", "al", "su", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
]

CATEGORICAL_FEATURES = [
    "rbc", "pc", "pcc", "ba", "htn", "dm",
    "cad", "appet", "pe", "ane"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ALLOWED_VALUES = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["present", "notpresent"],
    "ba": ["present", "notpresent"],
    "htn": ["yes", "no"],
    "dm": ["yes", "no"],
    "cad": ["yes", "no"],
    "appet": ["good", "poor"],
    "pe": ["yes", "no"],
    "ane": ["yes", "no"]
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "CKD Prediction API is running",
        "models": list(models.keys())
    })


@app.route("/api/models")
def get_models():
    return jsonify({
        "models": list(models.keys()),
        "default_model": default_model
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON request"}), 400

        selected_model = data.get("model", default_model)

        if selected_model not in models:
            return jsonify({"error": "Invalid model selected"}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing_features": missing
            }), 400

        input_data = {}

        for feature in NUMERIC_FEATURES:
            raw_value = data[feature]

            if raw_value is None or str(raw_value).strip() == "":
                input_data[feature] = np.nan
                continue

            try:
                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError
                input_data[feature] = value
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid numeric value for {feature}"
                }), 400

        for feature in CATEGORICAL_FEATURES:
            raw_value = data[feature]

            if raw_value is None or str(raw_value).strip() == "":
                input_data[feature] = np.nan
                continue

            value = str(raw_value).strip().lower()

            if value not in ALLOWED_VALUES[feature]:
                return jsonify({
                    "error": f"Invalid value for {feature}",
                    "allowed_values": ALLOWED_VALUES[feature]
                }), 400

            input_data[feature] = value

        input_df = pd.DataFrame([input_data], columns=FEATURES)

        pipeline = models[selected_model]
        prediction = int(pipeline.predict(input_df)[0])

        probability = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            classes = list(pipeline.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        return jsonify({
            "success": True,
            "model": selected_model,
            "prediction": prediction,
            "result": "CKD" if prediction == 1 else "Not CKD",
            "ckd_probability": (
                round(probability * 100, 2)
                if probability is not None else None
            ),
            "disclaimer": (
                "Educational machine-learning output only. "
                "This is not a medical diagnosis."
            )
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500



@app.route("/api/analytics")
def analytics():
    try:
        data_dir = os.path.join(BASE_DIR, "static", "data")

        performance = pd.read_csv(
            os.path.join(data_dir, "model_performance.csv")
        ).replace({np.nan: None}).to_dict(orient="records")

        cross_validation = pd.read_csv(
            os.path.join(data_dir, "cross_validation_results.csv")
        ).replace({np.nan: None}).to_dict(orient="records")

        importance_df = pd.read_csv(
            os.path.join(data_dir, "feature_importance.csv")
        )

        importance = {}
        for algorithm, group in importance_df.groupby("Algorithm"):
            top = group.sort_values(
                "Importance", ascending=False
            ).head(8)

            importance[algorithm] = top[
                ["Feature", "Importance"]
            ].to_dict(orient="records")

        return jsonify({
            "performance": performance,
            "cross_validation": cross_validation,
            "feature_importance": importance
        })

    except Exception:
        app.logger.exception("Analytics loading failed")
        return jsonify({"error": "Unable to load model analytics"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
