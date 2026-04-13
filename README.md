# 🧠 Brain Tumor Detection & Classification

An end-to-end web application for brain tumor detection and classification from MRI scans using a deep learning ensemble model with Grad-CAM visualisation.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)

---

## 🔍 Overview

The model classifies MRI brain scans into four categories:
| Class | Description |
|-------|-------------|
| **Glioma** | Tumor from glial cells in brain/spinal cord |
| **Meningioma** | Tumor from meninges membranes |
| **Pituitary Tumor** | Abnormal growth in the pituitary gland |
| **No Tumor** | Normal MRI scan |

---

## 🏗️ Architecture

### Model — Attention-Weighted Ensemble CNN
- **Custom CNN** — 4-layer conv network trained from scratch
- **MobileNetV2** — Pre-trained on ImageNet, fine-tuned
- **EfficientNetB0** — Pre-trained on ImageNet, fine-tuned
- **Attention Weighting** — Learnable weights that dynamically trust each branch

### Training Techniques
- Two-phase training (frozen → fine-tune last 30 layers)
- Label smoothing (0.10) to prevent overconfidence
- AdamW + Cosine LR schedule with warm-up
- Strong augmentation (rotation, zoom, flip, shear, channel shift)
- Class weighting for balanced training

### Visualisation — Grad-CAM
Generates heatmaps showing **which region of the MRI** the EfficientNetB0 branch focused on during classification. Red/yellow = highest model attention = likely tumour location.

---

## 📁 Project Structure

```
Brain-tumor-detection/
├── Ai/
│   ├── app.py                      # Flask REST API
│   ├── GRADCAM_heatmap.py          # Grad-CAM pipeline
│   ├── Cnn_model.py                # Training script
│   ├── class_indices.json          # Class name mappings
│   └── final_brain_tumor_model.keras  # Trained model weights
│
└── frontend/
    ├── src/
    │   ├── App.jsx                 # Main React component
    │   └── index.css               # Styling
    ├── index.html
    └── package.json
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Install Python dependencies
```bash
cd Ai
pip install flask flask-cors tensorflow opencv-python pillow
```

### 2. Start the Flask API
```bash
cd Ai
python app.py
# Runs on http://localhost:5000
```

### 3. Install and start the React frontend
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

### 4. Open the app
Visit **http://localhost:5173** and upload an MRI image.

---

## 🖼️ Usage

1. Upload an MRI brain scan (JPG, PNG, BMP, WEBP)
2. Click **Analyse MRI**
3. View:
   - **Tumour classification** with confidence %
   - **Class probability bars** for all 4 categories
   - **Grad-CAM heatmap** highlighting the tumour region

---

## 📊 Model Performance

| Phase | Best Val Accuracy |
|-------|------------------|
| Phase 1 (frozen backbone) | ~90–92% |
| Phase 2 (fine-tuned) | ~97–98% |

---

## 🔬 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns classes |
| `POST` | `/predict` | Upload MRI image → returns prediction + Grad-CAM |

### `/predict` Response
```json
{
  "prediction": "glioma",
  "display_label": "Glioma",
  "confidence": 0.928,
  "probabilities": { "glioma": 0.928, "meningioma": 0.025, ... },
  "original_b64": "<base64 PNG>",
  "overlay_b64":  "<base64 PNG>",
  "heatmap_b64":  "<base64 PNG>"
}
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + Vite |
| Backend | Python + Flask |
| ML Framework | TensorFlow / Keras 3 |
| Visualisation | Grad-CAM + OpenCV |
| Models | MobileNetV2, EfficientNetB0, Custom CNN |

---

## ⚠️ Note on Model File

The trained model (`final_brain_tumor_model.keras`, ~64 MB) is included in this repository. If you wish to retrain it, run:
```bash
cd Ai
python Cnn_model.py
```
> Requires the dataset in a `data/Training` and `data/Testing` folder structure.

---

## 👨‍💻 Author

**Himesh Gonnade**
