"""
app.py – Flask REST API for Brain Tumor Detection
Endpoints:
  POST /predict   — upload MRI image → returns prediction + Grad-CAM heatmaps
  GET  /health    — health check

FIX: Keras 3 cannot deserialize Lambda layers from .keras files.
     We rebuild the EXACT same architecture and extract model.weights.h5
     from the .keras ZIP to load weights by layer name — bypassing all
     graph-path mismatch errors.
"""

import os
import json
import traceback
import zipfile
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, Input, GlobalAveragePooling2D,
    Conv2D, MaxPooling2D, BatchNormalization,
    Lambda, Multiply, Reshape, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Local modules ──
from GRADCAM_heatmap import run_full_gradcam_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "final_brain_tumor_model.keras")
CLASS_IDX_PATH  = os.path.join(BASE_DIR, "class_indices.json")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CLASS INDICES
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CLASS_IDX_PATH):
    raise FileNotFoundError(
        f"class_indices.json not found at '{CLASS_IDX_PATH}'. "
        "Please run Cnn_model.py first."
    )

with open(CLASS_IDX_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {int(v): k for k, v in class_indices.items()}
CLASS_NAMES   = [idx_to_class[i] for i in range(len(idx_to_class))]
NUM_CLASSES   = len(CLASS_NAMES)
print(f"[INFO] Classes: {CLASS_NAMES}")

# ─────────────────────────────────────────────────────────────────────────────
# REBUILD MODEL ARCHITECTURE  (must exactly match Cnn_model.py)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Rebuilding model architecture …")

input_layer = Input(shape=(224, 224, 3), name="input_image")

# ── Branch A: Custom CNN ──────────────────────────────────────────────────
def build_custom_cnn(x):
    x = Conv2D(32,  (3,3), activation='relu', padding='same', name="cnn_c1")(x)
    x = BatchNormalization(name="cnn_bn1")(x)
    x = MaxPooling2D(2, 2, name="cnn_p1")(x)

    x = Conv2D(64,  (3,3), activation='relu', padding='same', name="cnn_c2")(x)
    x = BatchNormalization(name="cnn_bn2")(x)
    x = MaxPooling2D(2, 2, name="cnn_p2")(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same', name="cnn_c3")(x)
    x = BatchNormalization(name="cnn_bn3")(x)
    x = MaxPooling2D(2, 2, name="cnn_p3")(x)

    x = Conv2D(256, (3,3), activation='relu', padding='same', name="cnn_c4")(x)
    x = BatchNormalization(name="cnn_bn4")(x)
    x = GlobalAveragePooling2D(name="cnn_gap")(x)

    x = Dense(256, activation='relu',  name="cnn_d1")(x)
    x = Dropout(0.4, name="cnn_drop1")(x)
    x = Dense(128, activation='relu',  name="cnn_d2")(x)
    x = Dropout(0.3, name="cnn_drop2")(x)
    return Dense(NUM_CLASSES, activation='softmax', name="cnn_out")(x)

cnn_out = build_custom_cnn(input_layer)

# ── Branch B: MobileNetV2 ─────────────────────────────────────────────────
base_mobile = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))
for layer in base_mobile.layers:
    layer.trainable = False

mobile_x   = base_mobile(input_layer, training=False)
mobile_x   = GlobalAveragePooling2D(name="mobile_gap")(mobile_x)
mobile_x   = Dense(256, activation='relu',  name="mobile_d1")(mobile_x)
mobile_x   = Dropout(0.35, name="mobile_drop")(mobile_x)
mobile_out = Dense(NUM_CLASSES, activation='softmax', name="mobile_out")(mobile_x)

# ── Branch C: EfficientNetB0 ──────────────────────────────────────────────
base_eff = EfficientNetB0(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))
for layer in base_eff.layers:
    layer.trainable = False

eff_x   = base_eff(input_layer, training=False)
eff_x   = GlobalAveragePooling2D(name="eff_gap")(eff_x)
eff_x   = Dense(256, activation='relu',  name="eff_d1")(eff_x)
eff_x   = Dropout(0.35, name="eff_drop")(eff_x)
eff_out = Dense(NUM_CLASSES, activation='softmax', name="eff_out")(eff_x)

# ── Learned Weighted Ensemble ─────────────────────────────────────────────
stacked = Lambda(
    lambda tensors: tf.stack(tensors, axis=1),
    name="stack"
)([cnn_out, mobile_out, eff_out])   # (batch, 3, NUM_CLASSES)

attn_input = Concatenate(name="attn_in")([cnn_out, mobile_out, eff_out])
attn       = Dense(64, activation='relu',    name="attn_d1")(attn_input)
attn       = Dense(3,  activation='softmax', name="attn_weights")(attn)
attn       = Reshape((3, 1), name="attn_r")(attn)

weighted     = Multiply(name="weighted_stack")([stacked, attn])
ensemble_out = Lambda(
    lambda x: tf.reduce_sum(x, axis=1),
    name="ensemble_sum"
)(weighted)

model = Model(inputs=input_layer, outputs=ensemble_out, name="BrainTumorEnsemble_v2")
print("[INFO] Architecture rebuilt successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD WEIGHTS  (bypasses Keras 3 Lambda deserialization bug)
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'. "
        "Please run Cnn_model.py first to train and save the model."
    )

print("[INFO] Extracting and loading weights from .keras archive …")
_tmp_dir = tempfile.mkdtemp()
_weights_h5 = os.path.join(_tmp_dir, "model.weights.h5")
with zipfile.ZipFile(MODEL_PATH, "r") as _zf:
    with _zf.open("model.weights.h5") as _src, open(_weights_h5, "wb") as _dst:
        _dst.write(_src.read())
model.load_weights(_weights_h5)
print("[INFO] Weights loaded successfully.")

# Display-friendly labels
DISPLAY_LABELS = {
    "glioma"      : "Glioma",
    "meningioma"  : "Meningioma",
    "notumor"     : "No Tumor",
    "pituitary"   : "Pituitary Tumor"
}

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data with key 'file' (image).
    Returns JSON:
    {
      "prediction"   : "glioma",
      "display_label": "Glioma",
      "confidence"   : 0.97,
      "probabilities": { "glioma": 0.97, "meningioma": 0.01, ... },
      "original_b64" : "<base64 PNG>",
      "overlay_b64"  : "<base64 PNG>",
      "heatmap_b64"  : "<base64 PNG>"
    }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    try:
        img_bytes = file.read()

        # Run Grad-CAM pipeline (includes prediction)
        result = run_full_gradcam_pipeline(img_bytes, model)

        pred_idx   = result["pred_index"]
        probs      = result["probabilities"]
        pred_class = CLASS_NAMES[pred_idx]

        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}

        response = {
            "prediction"   : pred_class,
            "display_label": DISPLAY_LABELS.get(pred_class, pred_class.title()),
            "confidence"   : round(float(probs[pred_idx]), 6),
            "probabilities": prob_dict,
            "original_b64" : result["original_b64"],
            "overlay_b64"  : result["overlay_b64"],
            "heatmap_b64"  : result["heatmap_b64"]
        }
        return jsonify(response), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
