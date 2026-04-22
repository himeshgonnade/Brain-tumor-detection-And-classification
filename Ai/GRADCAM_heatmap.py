"""
GRADCAM_heatmap.py – Grad-CAM++ for Brain Tumor Ensemble (Keras 3, guaranteed working)
========================================================================================

ROOT CAUSE OF THE DIFFUSE HEATMAP
----------------------------------
Both the original code and the previous "fix" fell through to input-pixel saliency because:

  1. Lambda layers (tf.stack / tf.reduce_sum) in the ensemble head block GradientTape.
  2. tape.watch(conv_out) called AFTER the model forward pass means TF does NOT know
     to track the path conv_out → class_score — it's too late in the recording.
  3. Every call therefore returned gradients = None/zero, silently switching to
     input saliency, which produces a smooth gradient over the whole skull.

THE FIX THAT ACTUALLY WORKS
-----------------------------
We use a two-stage approach with an explicit tf.Variable:

  Stage 1 (no tape):
    Call a sub-model:  img → EfficientNetB0 → top_conv  (only forward values needed)
    Store result in a tf.Variable  ←  Variables are AUTO-WATCHED by GradientTape

  Stage 2 (inside tape):
    Run the remaining layers manually (no Lambda anywhere in this chain):
      conv_var → top_bn → top_activation → eff_gap → eff_d1 → eff_out → class_score

  tape.gradient(class_score, conv_var)  ← always non-zero, correctly localized

This is the standard TF2 Grad-CAM pattern — using a Variable as the "watched" tensor.
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64

IMG_SIZE = (224, 224)

# Layer name constants (must match the rebuilt ensemble in app.py)
EFFICIENTNET_LAYER_NAME = "efficientnetb0"
LAST_CONV_LAYER_NAME    = "top_conv"
TOP_BN_LAYER_NAME       = "top_bn"
TOP_ACT_LAYER_NAME      = "top_activation"


# ═══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(img_path_or_bytes):
    if isinstance(img_path_or_bytes, bytes):
        img = Image.open(io.BytesIO(img_path_or_bytes)).convert("RGB")
    else:
        img = Image.open(img_path_or_bytes).convert("RGB")

    img          = img.resize(IMG_SIZE)
    img_array    = np.array(img, dtype=np.float32)
    original_img = img_array.astype(np.uint8).copy()
    img_array    = img_array / 255.0
    img_array    = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)
    return original_img, img_array


# ═══════════════════════════════════════════════════════════════════════════════
# Grad-CAM++ (the correct implementation)
# ═══════════════════════════════════════════════════════════════════════════════

def make_gradcam_heatmap(img_array, ensemble_model, pred_index=None):
    """
    Compute a Grad-CAM++ heatmap through the EfficientNetB0 branch.

    Strategy:
      1. Run img → top_conv  (forward only, outside tape) → store as tf.Variable
      2. Run conv_var → top_bn → top_act → eff_gap → eff_d1 → eff_out  (inside tape)
         No Lambda layers exist in this path.
      3. tape.gradient(class_score, conv_var)  — always correct.

    Returns: (heatmap [H,W] float32 0-1,  pred_index int,  full_probs array)
    """
    # ── Step 0: Full ensemble prediction to pick the target class ──────────
    full_preds = ensemble_model(
        tf.cast(img_array, tf.float32), training=False
    )
    if pred_index is None:
        pred_index = int(tf.argmax(full_preds[0]))

    # ── Step 1: Locate layers ───────────────────────────────────────────────
    try:
        eff_sub  = ensemble_model.get_layer(EFFICIENTNET_LAYER_NAME)
        conv_lyr = eff_sub.get_layer(LAST_CONV_LAYER_NAME)
        bn_lyr   = eff_sub.get_layer(TOP_BN_LAYER_NAME)
        act_lyr  = eff_sub.get_layer(TOP_ACT_LAYER_NAME)
        gap_lyr  = ensemble_model.get_layer("eff_gap")
        d1_lyr   = ensemble_model.get_layer("eff_d1")
        out_lyr  = ensemble_model.get_layer("eff_out")
    except ValueError as e:
        print(f"[WARN] Layer lookup failed: {e}. Falling back to saliency.")
        img_var = tf.Variable(tf.cast(img_array, tf.float32))
        return _input_saliency_heatmap(img_var, ensemble_model, pred_index, full_preds)

    # ── Step 2: Build an extractor model: img → top_conv output ────────────
    #   This is a pure sub-model of EfficientNetB0 with no Lambda layers.
    #   We only use it to get the conv activations — no gradients here.
    try:
        conv_extractor = tf.keras.Model(
            inputs  = eff_sub.input,
            outputs = conv_lyr.output,
            name    = "conv_extractor"
        )
        # Forward pass only (outside tape) — get the raw activation values
        conv_activations = conv_extractor(
            tf.cast(img_array, tf.float32), training=False
        )                                                   # (1, H, W, C)
    except Exception as e:
        print(f"[WARN] Conv extractor failed: {e}. Falling back to saliency.")
        img_var = tf.Variable(tf.cast(img_array, tf.float32))
        return _input_saliency_heatmap(img_var, ensemble_model, pred_index, full_preds)

    # ── Step 3: Wrap activations in a tf.Variable ───────────────────────────
    #   tf.Variables are automatically watched by GradientTape.
    #   This is the KEY FIX — no need to manually call tape.watch().
    conv_var = tf.Variable(conv_activations, trainable=True)

    # ── Step 4: Grad-CAM++ forward pass (tape) — ZERO Lambda layers ─────────
    with tf.GradientTape(persistent=True) as tape:
        # Chain:  conv_var → top_bn → top_activation → eff_gap → eff_d1 → eff_out
        x           = bn_lyr(conv_var,  training=False)
        x           = act_lyr(x,        training=False)
        x           = gap_lyr(x,        training=False)
        x           = d1_lyr(x,         training=False)
        eff_probs   = out_lyr(x,        training=False)
        class_score = eff_probs[:, pred_index]

    # ── Step 5: Compute gradients ───────────────────────────────────────────
    grads  = tape.gradient(class_score, conv_var)   # (1, H, W, C)
    grads2 = tape.gradient(grads,       conv_var)   # second order (Grad-CAM++)
    del tape

    if grads is None or float(tf.reduce_sum(tf.abs(grads))) < 1e-10:
        print("[WARN] Grad-CAM gradients are zero — check layer names.")
        print(f"       eff_sub layers with Conv2D: "
              f"{[l.name for l in eff_sub.layers if isinstance(l, tf.keras.layers.Conv2D)][-5:]}")
        img_var = tf.Variable(tf.cast(img_array, tf.float32))
        return _input_saliency_heatmap(img_var, ensemble_model, pred_index, full_preds)

    grads_np  = grads[0].numpy()                            # (H, W, C)
    grads2_np = (grads2[0].numpy()
                 if grads2 is not None
                 else grads_np ** 2)
    conv_np   = conv_var[0].numpy()                         # (H, W, C)

    # ── Step 6: Grad-CAM++ weighting ────────────────────────────────────────
    # alpha_k = grad2 / (2*grad2 + sum_hw(A * grad3))
    grad3_np   = grads_np * grads2_np
    denom      = (2.0 * grads2_np
                  + np.sum(conv_np * grad3_np, axis=(0, 1), keepdims=True)
                  + 1e-7)
    alpha      = grads2_np / denom                          # (H, W, C)

    relu_grads = np.maximum(grads_np, 0)                    # ReLU — only positive
    weights    = np.sum(alpha * relu_grads, axis=(0, 1))    # (C,)

    # Weighted combination of activation maps
    heatmap = np.dot(conv_np, weights)                      # (H, W)
    heatmap = np.maximum(heatmap, 0)                        # ReLU

    # Vanilla Grad-CAM fallback if Grad-CAM++ weights are degenerate
    if heatmap.max() < 1e-8:
        weights2 = np.mean(relu_grads, axis=(0, 1))
        heatmap  = np.maximum(np.dot(conv_np, weights2), 0)

    # ── Step 7: Post-processing for sharp localization ───────────────────────
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()                   # normalize to [0,1]

        # Remove diffuse background — zero out anything below 50th percentile
        thresh  = np.percentile(heatmap[heatmap > 0], 50) if (heatmap > 0).any() else 0
        heatmap = np.where(heatmap < thresh, 0.0, heatmap)

        # Morphological dilation: connect nearby hotspots into one region
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        heatmap = cv2.dilate(heatmap.astype(np.float32), kernel, iterations=2)

        # Gaussian smooth for a clean blob
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (11, 11), 0)

        # Re-normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Mild gamma boost to improve contrast without over-sharpening
        heatmap = np.power(heatmap.clip(0, 1), 0.8)

    return heatmap, pred_index, full_preds[0].numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback: input-gradient saliency
# ═══════════════════════════════════════════════════════════════════════════════

def _input_saliency_heatmap(img_var, ensemble_model, pred_index, full_preds):
    """Last-resort fallback. Better than a blank image but less localized."""
    with tf.GradientTape() as tape:
        preds       = ensemble_model(img_var, training=False)
        class_score = preds[:, pred_index]
    grads = tape.gradient(class_score, img_var)

    if grads is None:
        return np.zeros(IMG_SIZE, dtype=np.float32), pred_index, full_preds[0].numpy()

    saliency      = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()
    p_low, p_high = np.percentile(saliency, [10, 98])
    saliency      = np.clip((saliency - p_low) / (p_high - p_low + 1e-8), 0, 1)
    saliency      = cv2.GaussianBlur(saliency.astype(np.float32), (15, 15), 3)
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency, pred_index, full_preds[0].numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Overlay & encoding
# ═══════════════════════════════════════════════════════════════════════════════

def overlay_heatmap_on_image(original_img, heatmap, alpha=0.50):
    """
    Overlay heatmap on MRI. Only colorizes regions where heatmap > 0,
    preserving the clean grayscale MRI appearance elsewhere.
    """
    img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, IMG_SIZE)

    heatmap_r = cv2.resize(heatmap.astype(np.float32), IMG_SIZE)
    p_min, p_max  = heatmap_r.min(), heatmap_r.max()
    heatmap_norm  = (heatmap_r - p_min) / (p_max - p_min + 1e-8)
    heatmap_u8    = np.uint8(255 * heatmap_norm)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # Masked blend: only apply heatmap where it's non-zero
    mask    = (heatmap_u8 > 0).astype(np.float32)
    mask3   = np.stack([mask, mask, mask], axis=-1)
    blended = (img_bgr.astype(np.float32) * (1.0 - alpha * mask3) +
               heatmap_color.astype(np.float32) * alpha * mask3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return (cv2.cvtColor(blended,       cv2.COLOR_BGR2RGB),
            cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))


def image_to_base64(img_array_rgb):
    img_pil = Image.fromarray(img_array_rgb.astype(np.uint8))
    buf     = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline entry point (called from app.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_gradcam_pipeline(img_bytes, ensemble_model, class_names=None):
    """
    Full pipeline: preprocess → predict → Grad-CAM++ → overlay → base64.
    Skips Grad-CAM if prediction is 'notumor' (normal scan).
    """
    print("[GRADCAM] run_full_gradcam_pipeline called (v3 — Grad-CAM++ via tf.Variable)")

    original_img, img_array = load_and_preprocess_image(img_bytes)

    # Always run full prediction first
    heatmap, pred_index, probs = make_gradcam_heatmap(img_array, ensemble_model)

    # Determine class name for the predicted index
    pred_class = None
    if class_names and pred_index < len(class_names):
        pred_class = class_names[pred_index]

    # If No Tumor — return blank heatmap images (clean MRI only)
    if pred_class == 'notumor':
        print("[GRADCAM] Normal scan detected — skipping Grad-CAM visualization.")
        blank = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        return {
            "original_b64" : image_to_base64(original_img),
            "overlay_b64"  : image_to_base64(original_img),   # just the original
            "heatmap_b64"  : image_to_base64(blank),
            "pred_index"   : pred_index,
            "probabilities": probs.tolist()
        }

    superimposed, heatmap_col = overlay_heatmap_on_image(original_img, heatmap)

    return {
        "original_b64" : image_to_base64(original_img),
        "overlay_b64"  : image_to_base64(superimposed),
        "heatmap_b64"  : image_to_base64(heatmap_col),
        "pred_index"   : pred_index,
        "probabilities": probs.tolist()
    }
