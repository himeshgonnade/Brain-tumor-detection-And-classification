"""
Brain Tumor Ensemble CNN – High-Accuracy Training Script
Target: 97-98% validation accuracy

Techniques used:
  1. TWO-PHASE training  (Phase-1: heads only → Phase-2: full fine-tune)
  2. Label Smoothing     (prevents overconfidence, better generalisation)
  3. Class Weights       (balances glioma/meningioma/notumor/pituitary)
  4. AdamW + Cosine Warmup LR schedule
  5. Strong augmentation (channel shift, shear, vertical flip)
  6. Gradient Clipping   (stabilises training)
  7. More unfreezing     (last 30 layers of each pretrained model in Phase-2)
  8. Attention-weighted ensemble (learnable weights instead of simple Average)
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, Average, Input, GlobalAveragePooling2D,
    Conv2D, MaxPooling2D, BatchNormalization,
    Lambda, Multiply, Reshape, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
)

print(f"[INFO] TensorFlow version : {tf.__version__}")

# =====================================================================
# 2. PATHS
# =====================================================================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "..", "data")
train_dir       = os.path.join(DATA_DIR, "Training")
test_dir        = os.path.join(DATA_DIR, "Testing")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "final_brain_tumor_model.keras")
CLASS_IDX_PATH  = os.path.join(BASE_DIR, "class_indices.json")

print(f"[INFO] Training dir  : {train_dir}")
print(f"[INFO] Testing dir   : {test_dir}")

# =====================================================================
# 3. HYPERPARAMETERS
# =====================================================================
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32          # larger batch = more stable gradients
PHASE1_EPOCHS = 8           # train heads only
PHASE2_EPOCHS = 25          # full fine-tune  (total = 33 max, ES stops early)

PHASE1_LR     = 1e-3        # higher LR for fresh head layers
PHASE2_LR     = 5e-5        # lower LR for careful fine-tuning
UNFREEZE_N    = 30          # last N layers to unfreeze in Phase-2

LABEL_SMOOTH  = 0.10        # label smoothing factor
CLIP_NORM     = 1.0         # gradient clipping

# =====================================================================
# 4. DATA GENERATORS  (strong augmentation on training set)
# =====================================================================
train_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    rotation_range   = 20,
    zoom_range       = 0.20,
    width_shift_range= 0.15,
    height_shift_range=0.15,
    horizontal_flip  = True,
    vertical_flip    = True,       # MRI scans can be flipped vertically too
    shear_range      = 0.10,
    brightness_range = [0.80, 1.20],
    channel_shift_range = 20.0,    # slight colour jitter
    fill_mode        = 'reflect'   # avoids black border artifacts
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = 'categorical',
    shuffle      = True,
    seed         = 42
)

test_data = val_datagen.flow_from_directory(
    test_dir,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = 'categorical',
    shuffle      = False
)

NUM_CLASSES = train_data.num_classes
CLASS_INDICES = train_data.class_indices
print(f"[INFO] Classes : {CLASS_INDICES}")

# Save class indices
with open(CLASS_IDX_PATH, "w") as f:
    json.dump(CLASS_INDICES, f)
print(f"[INFO] Class indices saved -> {CLASS_IDX_PATH}")

# =====================================================================
# 5. CLASS WEIGHTS  (handle dataset imbalance)
# =====================================================================
# Training counts: glioma=1321, meningioma=1339, notumor=1595, pituitary=1457
counts = {}
for cls, idx in CLASS_INDICES.items():
    cls_dir = os.path.join(train_dir, cls)
    counts[idx] = len([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

total_samples = sum(counts.values())
n_classes     = len(counts)
class_weights = {
    idx: total_samples / (n_classes * cnt)
    for idx, cnt in counts.items()
}
print(f"[INFO] Class counts  : {counts}")
print(f"[INFO] Class weights : { {k: round(v,3) for k,v in class_weights.items()} }")

# =====================================================================
# 6. MODEL ARCHITECTURE
# =====================================================================

input_layer = Input(shape=(224, 224, 3), name="input_image")

# ── Branch A: Custom CNN ─────────────────────────────────────────────
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

# ── Branch B: MobileNetV2 ────────────────────────────────────────────
base_mobile = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))
# Phase-1: freeze all
for layer in base_mobile.layers:
    layer.trainable = False

mobile_x   = base_mobile(input_layer, training=False)
mobile_x   = GlobalAveragePooling2D(name="mobile_gap")(mobile_x)
mobile_x   = Dense(256, activation='relu',  name="mobile_d1")(mobile_x)
mobile_x   = Dropout(0.35, name="mobile_drop")(mobile_x)
mobile_out = Dense(NUM_CLASSES, activation='softmax', name="mobile_out")(mobile_x)

# ── Branch C: EfficientNetB0 ─────────────────────────────────────────
base_eff = EfficientNetB0(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))
# Phase-1: freeze all
for layer in base_eff.layers:
    layer.trainable = False

eff_x   = base_eff(input_layer, training=False)
eff_x   = GlobalAveragePooling2D(name="eff_gap")(eff_x)
eff_x   = Dense(256, activation='relu',  name="eff_d1")(eff_x)
eff_x   = Dropout(0.35, name="eff_drop")(eff_x)
eff_out = Dense(NUM_CLASSES, activation='softmax', name="eff_out")(eff_x)

# ── Learned Weighted Ensemble ────────────────────────────────────────
# Trainable attention weights decide how much to trust each branch.
# Each branch output: (batch, NUM_CLASSES)
# Stack along axis=1 → (batch, 3, NUM_CLASSES)
stacked = Lambda(
    lambda tensors: tf.stack(tensors, axis=1),
    name="stack"
)([cnn_out, mobile_out, eff_out])   # (batch, 3, 4)

# Compute 3 scalar attention weights from concatenated branch outputs
attn_input = Concatenate(name="attn_in")([cnn_out, mobile_out, eff_out])  # (batch, 12)
attn       = Dense(64, activation='relu',    name="attn_d1")(attn_input)
attn       = Dense(3,  activation='softmax', name="attn_weights")(attn)    # (batch, 3)
attn       = Reshape((3, 1), name="attn_r")(attn)                          # (batch, 3, 1)

# Weighted average: (batch,3,4) * (batch,3,1) → sum over axis=1 → (batch, 4)
weighted     = Multiply(name="weighted_stack")([stacked, attn])            # (batch, 3, 4)
ensemble_out = Lambda(
    lambda x: tf.reduce_sum(x, axis=1),   # sum across 3 branches → (batch, 4)
    name="ensemble_sum"
)(weighted)

# ── Final Model ──────────────────────────────────────────────────────
model = Model(inputs=input_layer, outputs=ensemble_out, name="BrainTumorEnsemble_v2")

# =====================================================================
# 7. HELPER: LR WARMUP + COSINE DECAY SCHEDULE
# =====================================================================
def cosine_warmup_schedule(epoch, lr, warmup_epochs=3, base_lr=PHASE2_LR, min_lr=1e-7):
    """Used in Phase-2: warm up for `warmup_epochs`, then cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * ((epoch + 1) / warmup_epochs)
    progress  = (epoch - warmup_epochs) / max(1, PHASE2_EPOCHS - warmup_epochs)
    cos_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (base_lr - min_lr) * cos_decay

# =====================================================================
# 8. PHASE-1: TRAIN ONLY CLASSIFICATION HEADS
#    Pretrained base layers frozen → fast convergence of new head layers
# =====================================================================
print("\n" + "="*60)
print("  PHASE 1: Training classification heads (pretrained frozen)")
print("="*60)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics   = ['accuracy']
)

model.summary()

phase1_callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping  (monitor='val_accuracy', patience=4,
                    restore_best_weights=True, verbose=1),
]

history1 = model.fit(
    train_data,
    validation_data = test_data,
    epochs          = PHASE1_EPOCHS,
    class_weight    = class_weights,
    callbacks       = phase1_callbacks
)

print(f"\n[Phase-1 best val_acc]: "
      f"{max(history1.history['val_accuracy'])*100:.2f}%\n")

# =====================================================================
# 9. PHASE-2: UNFREEZE AND FINE-TUNE
#    Unfreeze last 30 layers of each pretrained model
#    Use low LR + cosine warmup to avoid destroying ImageNet features
# =====================================================================
print("="*60)
print(f"  PHASE 2: Fine-tuning (unfreezing last {UNFREEZE_N} layers each)")
print("="*60)

# Unfreeze last N layers of MobileNetV2
for layer in base_mobile.layers:
    layer.trainable = False
for layer in base_mobile.layers[-UNFREEZE_N:]:
    layer.trainable = True

# Unfreeze last N layers of EfficientNetB0
for layer in base_eff.layers:
    layer.trainable = False
for layer in base_eff.layers[-UNFREEZE_N:]:
    layer.trainable = True

total_trainable = sum(1 for l in model.layers if l.trainable)
print(f"[INFO] Trainable layers in Phase-2: {total_trainable}")

# Use AdamW (Adam + weight decay) for better regularisation
model.compile(
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate   = PHASE2_LR,
        weight_decay    = 1e-4,
        clipnorm        = CLIP_NORM,
    ),
    loss    = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics = ['accuracy']
)

phase2_callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping  (monitor='val_accuracy', patience=7,
                    restore_best_weights=True, verbose=1),
    LearningRateScheduler(
        lambda epoch, lr: cosine_warmup_schedule(epoch, lr),
        verbose=1
    ),
]

history2 = model.fit(
    train_data,
    validation_data = test_data,
    epochs          = PHASE2_EPOCHS,
    class_weight    = class_weights,
    callbacks       = phase2_callbacks
)

# =====================================================================
# 10. FINAL EVALUATION
# =====================================================================
print("\n" + "="*60)
print("  FINAL EVALUATION ON TEST SET")
print("="*60)

loss, acc = model.evaluate(test_data, verbose=1)
best_phase1 = max(history1.history['val_accuracy']) * 100
best_phase2 = max(history2.history['val_accuracy']) * 100
best_overall = max(best_phase1, best_phase2)

print(f"\n[RESULT] Phase-1 best val_accuracy : {best_phase1:.2f}%")
print(f"[RESULT] Phase-2 best val_accuracy : {best_phase2:.2f}%")
print(f"[RESULT] Overall best val_accuracy : {best_overall:.2f}%")
print(f"[RESULT] Final test accuracy       : {acc * 100:.2f}%")
print(f"[RESULT] Final test loss           : {loss:.4f}")
print(f"[INFO]   Model saved -> {MODEL_SAVE_PATH}")