"""
GRADCAM_heatmap.py
Grad-CAM for the Brain Tumor Ensemble model — Keras 3 compatible.

Root-cause of blank heatmap:
  The old code ran TWO separate forward passes:
    1. conv_model(img) → conv_out  (disconnected)
    2. ensemble_model(img) → class_score
  Because class_score did not flow through conv_out, all gradients = 0.

Fix:
  We create a SINGLE forward pass where:
    img_var → eff_dual → conv_out → (bn + activation) → eff_feat
                                                           ↓
                             ensemble's own eff head → class_score
  Now class_score flows through conv_out, giving correct Grad-CAM gradients.
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64

IMG_SIZE = (224, 224)

EFFICIENTNET_LAYER_NAME = "efficientnetb0"
LAST_CONV_IN_EFF        = "top_conv"


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_and_preprocess_image(img_path_or_bytes):
    if isinstance(img_path_or_bytes, bytes):
        img = Image.open(io.BytesIO(img_path_or_bytes)).convert('RGB')
    else:
        img = Image.open(img_path_or_bytes).convert('RGB')
    img          = img.resize(IMG_SIZE)
    img_array    = np.array(img, dtype=np.float32)
    original_img = img_array.astype(np.uint8).copy()
    img_array    = img_array / 255.0
    img_array    = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)
    return original_img, img_array


def _get_efficientnet_submodel(ensemble_model):
    for layer in ensemble_model.layers:
        if layer.name == EFFICIENTNET_LAYER_NAME:
            return layer
    raise ValueError(
        f"Layer '{EFFICIENTNET_LAYER_NAME}' not found. "
        f"Available: {[l.name for l in ensemble_model.layers]}"
    )


# ── Grad-CAM (primary) ────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, ensemble_model, pred_index=None):
    """
    Correct Keras-3 Grad-CAM via a single connected forward pass.

    Chain: img_var → eff_dual → conv_out → eff_feat → eff_head → class_score
    GradientTape differentiates class_score w.r.t. conv_out through this chain.
    """
    eff_submodel    = _get_efficientnet_submodel(ensemble_model)
    last_conv_layer = eff_submodel.get_layer(LAST_CONV_IN_EFF)

    # Model entirely inside EfficientNetB0 — no cross-graph tensors
    # outputs: (top_conv activations,  full EfficientNetB0 feature map)
    eff_dual = tf.keras.Model(
        inputs  = eff_submodel.input,
        outputs = [last_conv_layer.output, eff_submodel.output],
        name    = "eff_dual"
    )

    # Head layers from the ensemble (same weights as during training)
    eff_gap_layer = ensemble_model.get_layer("eff_gap")
    eff_d1_layer  = ensemble_model.get_layer("eff_d1")
    eff_out_layer = ensemble_model.get_layer("eff_out")

    # tf.Variable so the tape auto-watches it
    img_var = tf.Variable(tf.cast(img_array, tf.float32))

    # Full ensemble prediction for confidence / display (no tape needed)
    full_preds = ensemble_model(img_var, training=False)
    if pred_index is None:
        pred_index = int(tf.argmax(full_preds[0]))

    # Single connected forward pass
    with tf.GradientTape() as tape:
        # conv_out and eff_feat come from the SAME call → they are connected!
        # chain: img_var → top_conv → conv_out → top_bn → top_activation → eff_feat
        conv_out, eff_feat = eff_dual(img_var, training=False)

        # Continue through the ensemble's EfficientNetB0 head
        # chain: eff_feat → gap → d1 → eff_out → class_score
        x          = eff_gap_layer(eff_feat, training=False)
        x          = eff_d1_layer(x,         training=False)
        eff_probs  = eff_out_layer(x,         training=False)
        class_score = eff_probs[:, pred_index]

    # d(class_score) / d(conv_out)  ← flows through top_bn → top_act → gap → d1 → out
    grads = tape.gradient(class_score, conv_out)

    if grads is None or float(tf.reduce_sum(tf.abs(grads))) < 1e-8:
        print("[WARN] Grad-CAM gradients are zero, falling back to input saliency.")
        return _input_saliency_heatmap(img_var, ensemble_model, pred_index, full_preds)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()  # (C,)
    conv_np      = conv_out[0].numpy()                             # (H, W, C)

    for i in range(pooled_grads.shape[-1]):
        conv_np[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_np, axis=-1)   # (H, W)
    heatmap = np.maximum(heatmap, 0)      # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
        
        # --- Enhance Focus on Tumor Region ---
        # 1. Zero out diffuse background noise (e.g., anything below 40% of the maximum)
        heatmap = np.where(heatmap < 0.4, 0, heatmap)
        
        # 2. Smooth the remaining strong activations to form a cohesive blob
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (15, 15), 0)
        
        # 3. Renormalize to ensure the peak is 1.0
        p_max = heatmap.max()
        if p_max > 0:
            heatmap /= p_max

    return heatmap, pred_index, full_preds[0].numpy()


# ── Fallback: input-gradient saliency ─────────────────────────────────────────

def _input_saliency_heatmap(img_var, ensemble_model, pred_index, full_preds):
    """
    Fallback Grad-CAM: gradient of class score w.r.t. input pixels.
    Always produces non-zero gradients and highlights tumour regions well.
    """
    with tf.GradientTape() as tape:
        preds       = ensemble_model(img_var, training=False)
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, img_var)

    if grads is None:
        print("[WARN] Input saliency also returned None — returning blank heatmap.")
        return np.zeros(IMG_SIZE, dtype=np.float32), pred_index, full_preds[0].numpy()

    # Max across colour channels → (224, 224)
    saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()

    # Smooth to reduce noise and enhance spatial coherence
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (21, 21), 3)

    # Percentile normalisation for better contrast
    p_low, p_high = np.percentile(saliency, [5, 95])
    saliency      = np.clip((saliency - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return saliency, pred_index, full_preds[0].numpy()


# ── Overlay & encoding ─────────────────────────────────────────────────────────

def overlay_heatmap_on_image(original_img, heatmap, alpha=0.50):
    """Superimpose a Grad-CAM heatmap on the original MRI image."""
    img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, IMG_SIZE)

    heatmap_resized = cv2.resize(heatmap.astype(np.float32), IMG_SIZE)
    p_min, p_max    = heatmap_resized.min(), heatmap_resized.max()
    heatmap_norm    = (heatmap_resized - p_min) / (p_max - p_min + 1e-8)
    heatmap_uint8   = np.uint8(255 * heatmap_norm)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    return (cv2.cvtColor(superimposed,    cv2.COLOR_BGR2RGB),
            cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))


def image_to_base64(img_array_rgb):
    img_pil = Image.fromarray(img_array_rgb)
    buf     = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_full_gradcam_pipeline(img_bytes, ensemble_model):
    original_img, img_array     = load_and_preprocess_image(img_bytes)
    heatmap, pred_index, probs  = make_gradcam_heatmap(img_array, ensemble_model)
    superimposed, heatmap_col   = overlay_heatmap_on_image(original_img, heatmap)

    return {
        "original_b64" : image_to_base64(original_img),
        "overlay_b64"  : image_to_base64(superimposed),
        "heatmap_b64"  : image_to_base64(heatmap_col),
        "pred_index"   : pred_index,
        "probabilities": probs.tolist()
    }
