from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import librosa
import numpy as np
import joblib
import time
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

# =====================================================================
# 🔧 CONFIG & SETUP
# =====================================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# =====================================================================
# 🎤 Load Speech Recognition Assets
# =====================================================================
speech_model = joblib.load("./Speech-Recognition/models_output/best_model.joblib")
speech_scaler = joblib.load("./Speech-Recognition/models_output/scaler.joblib")
speech_encoder = joblib.load("./Speech-Recognition/models_output/label_encoder.joblib")

latest_audio_pred = "none"

# =====================================================================
# 👁️ Load Computer Vision Model (PyTorch)
# =====================================================================
print("🔄 Loading Vision AI Model...")

vision_encoder = joblib.load("./Computer-Vision/vision_label_encoder.joblib")
num_classes = len(vision_encoder.classes_)

vision_model = models.resnet18(weights=None)
vision_model.fc = torch.nn.Linear(vision_model.fc.in_features, num_classes)
vision_model.load_state_dict(torch.load("./Computer-Vision/cheating_cnn_model.pth", map_location=DEVICE))
vision_model.to(DEVICE)
vision_model.eval()

cv_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

latest_vision_pred = "none"

# =====================================================================
# 🎤 SPEECH FEATURE EXTRACTOR
# =====================================================================
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return np.hstack([rms, zcr, spec, mfcc_mean])


# =====================================================================
# 🎤 AUDIO ENDPOINT
# =====================================================================
@app.post("/upload_audio")
async def upload_audio(file: UploadFile):
    global latest_audio_pred

    contents = await file.read()
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        feats = extract_features(temp_path).reshape(1, -1)
        feats_scaled = speech_scaler.transform(feats)
        pred = speech_model.predict(feats_scaled)[0]
        label = speech_encoder.inverse_transform([pred])[0]

        latest_audio_pred = label
        print("🎤 Audio Prediction:", label)

        return {"status": "ok", "prediction": label}

    except Exception as e:
        return {"status": "error", "msg": str(e)}


@app.get("/latest_audio")
def get_latest_audio():
    return {"label": latest_audio_pred, "timestamp": time.time()}


# =====================================================================
# 👁️ COMPUTER VISION ENDPOINT
# =====================================================================
@app.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    global latest_vision_pred

    try:
        # Read image into PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Transform
        img_tensor = cv_transform(image).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            outputs = vision_model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_label = vision_encoder.inverse_transform([pred_class])[0]

        latest_vision_pred = pred_label
        print("👁️ Vision Prediction:", pred_label)

        return {
            "status": "ok",
            "vision_prediction": pred_label
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/latest_vision")
def get_latest_vision():
    return {"label": latest_vision_pred, "timestamp": time.time()}


# =====================================================================
# SERVER RUN
# =====================================================================
if __name__ == "__main__":
    print("🚀 FastAPI running...")
    uvicorn.run(app, host="0.0.0.0", port=5000)