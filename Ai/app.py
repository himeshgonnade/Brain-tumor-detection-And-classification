"""
app.py – Flask REST API for Brain Tumor Detection
Endpoints:
  POST /predict   — upload MRI image → returns prediction + Grad-CAM heatmaps
  GET  /health    — health check (always fast)
  GET  /status    — model loading status (ready / loading / error)

FIX: Model is loaded in a background thread so Flask is immediately
     responsive. The 'failed to fetch' error was caused by Flask blocking
     ALL incoming connections while the heavy model-build ran on startup
     (downloading MobileNetV2 + EfficientNetB0 weights, rebuilding arch.).
"""

import os
import json
import traceback
import zipfile
import tempfile
import threading
import datetime
import uuid
import jwt
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
DB_PATH         = os.path.join(BASE_DIR, "mock_db.json")

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP  (created immediately — before model loads)
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-doctor-key'
CORS(app, resources={r"/*": {"origins": "*"}})   # allow all origins during dev

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

def init_db():
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, 'w') as f:
            json.dump({"patients": [], "reports": []}, f)

def read_db():
    if not os.path.exists(DB_PATH):
        init_db()
    try:
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"patients": [], "reports": []}

def write_db(data):
    with open(DB_PATH, 'w') as f:
        json.dump(data, f, indent=2)

init_db()

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL STATE
# ─────────────────────────────────────────────────────────────────────────────
model        = None
CLASS_NAMES  = []
NUM_CLASSES  = 0
_load_error  = None
_model_ready = threading.Event()   # set() once model is loaded

DISPLAY_LABELS = {
    "glioma"     : "Glioma",
    "meningioma" : "Meningioma",
    "notumor"    : "No Tumor",
    "pituitary"  : "Pituitary Tumor"
}

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
def _load_model_background():
    global model, CLASS_NAMES, NUM_CLASSES, _load_error

    try:
        # ── Load class indices ──────────────────────────────────────────────
        if not os.path.exists(CLASS_IDX_PATH):
            raise FileNotFoundError(
                f"class_indices.json not found at '{CLASS_IDX_PATH}'. "
                "Please run Cnn_model.py first."
            )
        with open(CLASS_IDX_PATH, "r") as f:
            class_indices = json.load(f)

        idx_to_class = {int(v): k for k, v in class_indices.items()}
        CLASS_NAMES  = [idx_to_class[i] for i in range(len(idx_to_class))]
        NUM_CLASSES  = len(CLASS_NAMES)
        print(f"[INFO] Classes: {CLASS_NAMES}")

        # ── Rebuild architecture ────────────────────────────────────────────
        print("[INFO] Rebuilding model architecture …")
        input_layer = Input(shape=(224, 224, 3), name="input_image")

        # Branch A: Custom CNN
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

        # Branch B: MobileNetV2
        base_mobile = MobileNetV2(weights='imagenet', include_top=False,
                                   input_shape=(224, 224, 3))
        for layer in base_mobile.layers:
            layer.trainable = False
        mobile_x   = base_mobile(input_layer, training=False)
        mobile_x   = GlobalAveragePooling2D(name="mobile_gap")(mobile_x)
        mobile_x   = Dense(256, activation='relu',  name="mobile_d1")(mobile_x)
        mobile_x   = Dropout(0.35, name="mobile_drop")(mobile_x)
        mobile_out = Dense(NUM_CLASSES, activation='softmax', name="mobile_out")(mobile_x)

        # Branch C: EfficientNetB0
        base_eff = EfficientNetB0(weights='imagenet', include_top=False,
                                   input_shape=(224, 224, 3))
        for layer in base_eff.layers:
            layer.trainable = False
        eff_x   = base_eff(input_layer, training=False)
        eff_x   = GlobalAveragePooling2D(name="eff_gap")(eff_x)
        eff_x   = Dense(256, activation='relu',  name="eff_d1")(eff_x)
        eff_x   = Dropout(0.35, name="eff_drop")(eff_x)
        eff_out = Dense(NUM_CLASSES, activation='softmax', name="eff_out")(eff_x)

        # Learned Weighted Ensemble
        stacked = Lambda(
            lambda tensors: tf.stack(tensors, axis=1),
            name="stack"
        )([cnn_out, mobile_out, eff_out])

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

        # ── Load weights ────────────────────────────────────────────────────
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{MODEL_PATH}'. "
                "Please run Cnn_model.py first to train and save the model."
            )

        print("[INFO] Extracting and loading weights from .keras archive …")
        _tmp_dir    = tempfile.mkdtemp()
        _weights_h5 = os.path.join(_tmp_dir, "model.weights.h5")
        with zipfile.ZipFile(MODEL_PATH, "r") as _zf:
            with _zf.open("model.weights.h5") as _src, open(_weights_h5, "wb") as _dst:
                _dst.write(_src.read())
        model.load_weights(_weights_h5)
        print("[INFO] Weights loaded successfully. Model is READY.")

    except Exception as exc:
        _load_error = str(exc)
        traceback.print_exc()
        print(f"[ERROR] Model failed to load: {exc}")
    finally:
        _model_ready.set()   # always signal — even on error


# Start background loader immediately
threading.Thread(target=_load_model_background, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

def token_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            token = token.split(" ")[1] # Bearer Token
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except Exception:
            return jsonify({'message': 'Token is invalid or expired!'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing credentials'}), 400
    
    # Mock authentication details
    if data['email'] == 'doctor@hospital.com' and data['password'] == 'password123':
        token = jwt.encode({
            'user': data['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token})
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route("/api/stats", methods=["GET"])
@token_required
def get_stats():
    db = read_db()
    patients_count = len(db.get("patients", []))
    reports = db.get("reports", [])
    tumors_detected = sum(1 for r in reports if r.get('prediction', '').lower() != 'notumor')
    normal_cases = sum(1 for r in reports if r.get('prediction', '').lower() == 'notumor')
    
    all_activity = sorted(reports, key=lambda x: x.get('date', ''), reverse=True)
    for act in all_activity:
        pt = next((p for p in db.get("patients", []) if p.get("id") == act.get("patient_id")), None)
        act["patient_name"] = pt["name"] if pt else "Unknown"
    
    return jsonify({
        "total_patients": patients_count,
        "tumors_detected": tumors_detected,
        "normal_cases": normal_cases,
        "recent_activity": all_activity
    })

@app.route("/api/patients", methods=["GET", "POST"])
@token_required
def manage_patients():
    db = read_db()
    if request.method == "GET":
        patients = sorted(db.get("patients", []), key=lambda x: x.get('created_at', ''), reverse=True)
        return jsonify(patients)
    
    if request.method == "POST":
        data = request.get_json()
        new_patient = {
            "id": str(uuid.uuid4())[:8],
            "name": data.get("name", "Unknown"),
            "age": data.get("age", 0),
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        db.setdefault("patients", []).append(new_patient)
        write_db(db)
        return jsonify(new_patient), 201

@app.route("/api/reports", methods=["GET", "POST"])
@token_required
def manage_reports():
    db = read_db()
    if request.method == "GET":
        patient_id = request.args.get("patient_id")
        reports = db.get("reports", [])
        if patient_id:
            reports = [r for r in reports if r.get("patient_id") == patient_id]
        
        # Enrich with patient name
        for r in reports:
            pt = next((p for p in db.get("patients", []) if p.get("id") == r.get("patient_id")), None)
            r["patient_name"] = pt["name"] if pt else "Unknown"
        
        reports = sorted(reports, key=lambda x: x.get('date', ''), reverse=True)
        return jsonify(reports)

    if request.method == "POST":
        data = request.get_json()
        new_report = {
            "id": str(uuid.uuid4())[:8],
            "patient_id": data.get("patient_id"),
            "prediction": data.get("prediction"),
            "confidence": data.get("confidence"),
            "heatmap_b64": data.get("heatmap_b64"),
            "overlay_b64": data.get("overlay_b64"),
            "original_b64": data.get("original_b64"),
            "date": datetime.datetime.utcnow().isoformat()
        }
        db.setdefault("reports", []).append(new_report)
        write_db(db)
        return jsonify(new_report), 201

@app.route("/health", methods=["GET"])
def health():
    """Always-fast health check — does NOT wait for model."""
    return jsonify({
        "status"      : "ok",
        "model_ready" : _model_ready.is_set() and _load_error is None,
        "classes"     : CLASS_NAMES
    })


@app.route("/status", methods=["GET"])
def status():
    """Poll this to know when the model is ready."""
    if not _model_ready.is_set():
        return jsonify({"ready": False, "message": "Model is loading, please wait…"}), 202
    if _load_error:
        return jsonify({"ready": False, "error": _load_error}), 500
    return jsonify({"ready": True, "classes": CLASS_NAMES}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data with key 'file' (image).
    Returns JSON with prediction + Grad-CAM base64 images.
    """
    # Immediately reject if model not ready
    if not _model_ready.is_set():
        return jsonify({"error": "Model is still loading. Please wait a moment and try again."}), 503
    if _load_error:
        return jsonify({"error": f"Model failed to load: {_load_error}"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    try:
        img_bytes = file.read()
        result    = run_full_gradcam_pipeline(img_bytes, model, class_names=CLASS_NAMES)

        pred_idx   = result["pred_index"]
        probs      = result["probabilities"]
        pred_class = CLASS_NAMES[pred_idx]
        prob_dict  = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}

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
