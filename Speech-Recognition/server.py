from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import uvicorn
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML assets
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("label_encoder.joblib")

latest_prediction = "none"

def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return np.hstack([rms, zcr, spec, mfcc_mean])

@app.post("/upload")
async def upload_audio(file: UploadFile):
    global latest_prediction

    contents = await file.read()
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        feats = extract_features(temp_path).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        pred = model.predict(feats_scaled)[0]
        label = encoder.inverse_transform([pred])[0]

        latest_prediction = label
        print("Prediction:", label)

        return {"status": "ok", "prediction": label}

    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.get("/latest")
def get_latest():
    return {"label": latest_prediction, "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
