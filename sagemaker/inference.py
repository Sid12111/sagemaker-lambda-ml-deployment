# Handlers used by SageMaker SKLearn container
# Accepts either application/json with {"instances": [[...], ...]} or text/csv

import io
import json
import os
import glob
import numpy as np
import joblib

def model_fn(model_dir):
    # Load your model from the SageMaker model directory
    candidates = [
        os.path.join(model_dir, "model.pkl"),
        os.path.join(model_dir, "model.joblib"),
        os.path.join(model_dir, "sample_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)

    pkl = glob.glob(os.path.join(model_dir, "*.pkl"))
    job = glob.glob(os.path.join(model_dir, "*.joblib"))
    files = pkl + job
    if not files:
        raise FileNotFoundError("No model file found in model_dir.")
    return joblib.load(files[0])

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            # Support {"instances": [...]}, {"inputs": [...]}, or {"input": [...]}
            for key in ("instances", "inputs", "input"):
                if key in data:
                    arr = np.array(data[key], dtype=float)
                    return arr
            # If it's a dict but no known keys, try value list
            arr = np.array(list(data.values()), dtype=float)
            return arr
        # If it's a list already
        return np.array(data, dtype=float)

    if content_type == "text/csv":
        # One or more CSV lines
        stream = io.StringIO(request_body)
        rows = []
        for line in stream:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(",")])
        return np.array(rows, dtype=float)

    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, accept):
    result = {"predictions": prediction.tolist()}
    return json.dumps(result), "application/json"
