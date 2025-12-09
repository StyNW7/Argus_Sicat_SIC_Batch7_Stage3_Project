# fastapi_server_cv_audio.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import uvicorn
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image
import io
import base64
import json
from datetime import datetime
import os
import warnings
import logging
from typing import Optional
import sqlite3

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# =====================================================================
# LOAD MODELS
# =====================================================================
print("🔧 Loading AI models...")

# Load Speech Recognition models
try:
    speech_model = joblib.load("./Speech-Recognition/models_output/best_model.joblib")
    speech_scaler = joblib.load("./Speech-Recognition/models_output/scaler.joblib")
    speech_encoder = joblib.load("./Speech-Recognition/models_output/label_encoder.joblib")
    print("✅ Speech Recognition models loaded")
except Exception as e:
    print(f"❌ Error loading Speech Recognition models: {e}")
    speech_model = speech_scaler = speech_encoder = None

# Load Computer Vision models
try:
    # Load label encoder
    cv_label_encoder = torch.load("./Computer-Vision/label_encoder.pt", map_location=DEVICE)
    
    # Load model architecture
    class CheatingCNN(nn.Module):
        def __init__(self, num_classes):
            super(CheatingCNN, self).__init__()
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
            
        def forward(self, x):
            return self.resnet(x)
    
    # Initialize model
    cv_model = CheatingCNN(num_classes=len(cv_label_encoder.classes_))
    cv_model.load_state_dict(torch.load("./Computer-Vision/cheating_cnn_model.pth", map_location=DEVICE))
    cv_model.to(DEVICE)
    cv_model.eval()
    print("✅ Computer Vision model loaded")
    print(f"   Classes: {cv_label_encoder.classes_}")
except Exception as e:
    print(f"❌ Error loading Computer Vision model: {e}")
    cv_model = cv_label_encoder = None

# Image transformations for CV
cv_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================================================================
# GLOBAL STATE
# =====================================================================
latest_predictions = {
    "speech": {"label": "none", "timestamp": time.time()},
    "vision": {"label": "none", "timestamp": time.time()}
}

device_history = []
integrity_score_history = []

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def extract_audio_features(audio_bytes):
    """Extract features from audio bytes"""
    try:
        # Save to temporary file
        temp_path = f"temp_audio_{int(time.time())}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Extract features
        y, sr = librosa.load(temp_path, sr=16000)
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spec = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Clean up
        os.remove(temp_path)
        
        return np.hstack([rms, zcr, spec, mfcc_mean])
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        raise

def predict_audio(audio_bytes):
    """Predict audio label"""
    if speech_model is None:
        return {"error": "Speech model not loaded"}
    
    try:
        features = extract_audio_features(audio_bytes).reshape(1, -1)
        features_scaled = speech_scaler.transform(features)
        pred = speech_model.predict(features_scaled)[0]
        label = speech_encoder.inverse_transform([pred])[0]
        
        # Get probabilities
        if hasattr(speech_model, 'predict_proba'):
            probabilities = speech_model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 0.8  # Default confidence
        
        return {
            "label": label,
            "confidence": confidence,
            "class_index": int(pred)
        }
    except Exception as e:
        logger.error(f"Error predicting audio: {e}")
        return {"error": str(e)}

def predict_image(image_bytes):
    """Predict image label"""
    if cv_model is None:
        return {"error": "CV model not loaded"}
    
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = cv_transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = cv_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_idx = predicted.item()
            confidence_val = confidence.item()
            
            # Get label
            label = cv_label_encoder.inverse_transform([predicted_idx])[0]
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            
        return {
            "label": label,
            "confidence": float(confidence_val),
            "class_index": int(predicted_idx),
            "probabilities": all_probs.tolist()
        }
    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        return {"error": str(e)}

def calculate_integrity_score(speech_pred, vision_pred, speech_weight=0.4, vision_weight=0.6):
    """Calculate integrity score based on predictions"""
    # Base score
    base_score = 100
    
    # Penalties based on predictions
    penalty = 0
    
    # Speech penalties
    if speech_pred.get("label") == "whispering":
        penalty += 2 * speech_weight
    elif speech_pred.get("label") == "normal_conversation":
        penalty += 1 * speech_weight
    
    # Vision penalties (adjust based on your labels)
    vision_label = vision_pred.get("label", "").lower()
    if "suspicious" in vision_label or "cheating" in vision_label:
        penalty += 5 * vision_weight
    elif "looking_away" in vision_label or "distracted" in vision_label:
        penalty += 3 * vision_weight
    elif "head_down" in vision_label:
        penalty += 2 * vision_weight
    
    # Calculate final score
    final_score = base_score - (penalty * 10)
    
    # Ensure score is within bounds
    return max(0, min(100, final_score))

def get_risk_level(score):
    """Determine risk level based on integrity score"""
    if score >= 70:
        return "Safe", "🟢", "success"
    elif score >= 35:
        return "Alert", "🟡", "warning"
    else:
        return "Warning", "🔴", "error"

# =====================================================================
# DATABASE SETUP
# =====================================================================
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('argus_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        prediction_type TEXT,
        label TEXT,
        confidence REAL,
        device_id TEXT,
        student_id TEXT,
        integrity_score REAL,
        risk_level TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        alert_type TEXT,
        severity TEXT,
        description TEXT,
        device_id TEXT,
        resolved BOOLEAN DEFAULT 0
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_database()

def save_prediction(prediction_type, label, confidence, device_id="unknown", student_id="unknown"):
    """Save prediction to database"""
    try:
        # Calculate integrity score
        integrity_score = calculate_integrity_score(
            latest_predictions["speech"],
            latest_predictions["vision"]
        )
        risk_level, _, _ = get_risk_level(integrity_score)
        
        conn = sqlite3.connect('argus_predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO predictions 
        (prediction_type, label, confidence, device_id, student_id, integrity_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (prediction_type, label, confidence, device_id, student_id, integrity_score, risk_level))
        
        # Check if this is an alert
        if label.lower() in ["whispering", "suspicious", "cheating"] and confidence > 0.7:
            cursor.execute('''
            INSERT INTO alerts (alert_type, severity, description, device_id)
            VALUES (?, ?, ?, ?)
            ''', (
                f"{prediction_type}_{label}",
                "high" if confidence > 0.8 else "medium",
                f"{label} detected with {confidence:.2f} confidence",
                device_id
            ))
        
        conn.commit()
        conn.close()
        
        # Update history
        integrity_score_history.append({
            "timestamp": time.time(),
            "score": integrity_score,
            "risk_level": risk_level
        })
        
        # Keep only last 100 entries
        if len(integrity_score_history) > 100:
            integrity_score_history.pop(0)
            
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")

# =====================================================================
# FASTAPI APP
# =====================================================================
app = FastAPI(
    title="Argus AI Server",
    description="Combined Computer Vision and Speech Recognition for Exam Monitoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# API ENDPOINTS
# =====================================================================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Argus AI Server",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "speech_recognition": speech_model is not None,
            "computer_vision": cv_model is not None
        },
        "device": DEVICE
    }

@app.post("/predict/audio")
async def predict_audio_endpoint(
    file: UploadFile = File(...),
    device_id: Optional[str] = None,
    student_id: Optional[str] = None
):
    """Predict audio from uploaded file"""
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Predict
        result = predict_audio(contents)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Update latest prediction
        latest_predictions["speech"] = {
            "label": result["label"],
            "confidence": result["confidence"],
            "timestamp": time.time()
        }
        
        # Save to database
        save_prediction(
            "speech",
            result["label"],
            result["confidence"],
            device_id or "unknown",
            student_id or "unknown"
        )
        
        return {
            "status": "success",
            "prediction": result["label"],
            "confidence": result["confidence"],
            "class_index": result.get("class_index"),
            "timestamp": time.time(),
            "device_id": device_id,
            "student_id": student_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image_endpoint(
    file: UploadFile = File(...),
    device_id: Optional[str] = None,
    student_id: Optional[str] = None
):
    """Predict image from uploaded file"""
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Predict
        result = predict_image(contents)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Update latest prediction
        latest_predictions["vision"] = {
            "label": result["label"],
            "confidence": result["confidence"],
            "timestamp": time.time(),
            "probabilities": result.get("probabilities", [])
        }
        
        # Save to database
        save_prediction(
            "vision",
            result["label"],
            result["confidence"],
            device_id or "unknown",
            student_id or "unknown"
        )
        
        return {
            "status": "success",
            "prediction": result["label"],
            "confidence": result["confidence"],
            "class_index": result.get("class_index"),
            "probabilities": result.get("probabilities", []),
            "timestamp": time.time(),
            "device_id": device_id,
            "student_id": student_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/both")
async def predict_both_endpoint(
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    device_id: Optional[str] = None,
    student_id: Optional[str] = None
):
    """Predict both audio and image"""
    results = {}
    
    # Process audio if provided
    if audio_file:
        try:
            audio_contents = await audio_file.read()
            audio_result = predict_audio(audio_contents)
            if "error" not in audio_result:
                latest_predictions["speech"] = {
                    "label": audio_result["label"],
                    "confidence": audio_result["confidence"],
                    "timestamp": time.time()
                }
                save_prediction("speech", audio_result["label"], audio_result["confidence"], device_id, student_id)
            results["audio"] = audio_result
        except Exception as e:
            results["audio"] = {"error": str(e)}
    
    # Process image if provided
    if image_file:
        try:
            image_contents = await image_file.read()
            image_result = predict_image(image_contents)
            if "error" not in image_result:
                latest_predictions["vision"] = {
                    "label": image_result["label"],
                    "confidence": image_result["confidence"],
                    "timestamp": time.time(),
                    "probabilities": image_result.get("probabilities", [])
                }
                save_prediction("vision", image_result["label"], image_result["confidence"], device_id, student_id)
            results["image"] = image_result
        except Exception as e:
            results["image"] = {"error": str(e)}
    
    # Calculate integrity score
    integrity_score = calculate_integrity_score(
        latest_predictions["speech"],
        latest_predictions["vision"]
    )
    risk_level, risk_emoji, _ = get_risk_level(integrity_score)
    
    return {
        "status": "success",
        "audio": results.get("audio"),
        "vision": results.get("image"),
        "integrity_score": integrity_score,
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "timestamp": time.time()
    }

@app.get("/latest")
async def get_latest():
    """Get latest predictions from both models"""
    # Calculate current integrity score
    integrity_score = calculate_integrity_score(
        latest_predictions["speech"],
        latest_predictions["vision"]
    )
    risk_level, risk_emoji, _ = get_risk_level(integrity_score)
    
    return {
        "speech": latest_predictions["speech"],
        "vision": latest_predictions["vision"],
        "integrity_score": integrity_score,
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "timestamp": time.time()
    }

@app.get("/history")
async def get_history(limit: int = 50):
    """Get prediction history"""
    try:
        conn = sqlite3.connect('argus_predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        history = []
        for row in rows:
            history.append(dict(zip(columns, row)))
        
        conn.close()
        
        return {
            "status": "success",
            "count": len(history),
            "history": history
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/alerts")
async def get_alerts(limit: int = 20, resolved: bool = False):
    """Get recent alerts"""
    try:
        conn = sqlite3.connect('argus_predictions.db')
        cursor = conn.cursor()
        
        if resolved:
            cursor.execute('''
            SELECT * FROM alerts 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
            SELECT * FROM alerts 
            WHERE resolved = 0 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        alerts = []
        for row in rows:
            alerts.append(dict(zip(columns, row)))
        
        conn.close()
        
        return {
            "status": "success",
            "count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    try:
        conn = sqlite3.connect('argus_predictions.db')
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_preds = cursor.fetchone()[0]
        
        # Predictions by type
        cursor.execute("SELECT prediction_type, COUNT(*) FROM predictions GROUP BY prediction_type")
        preds_by_type = dict(cursor.fetchall())
        
        # Predictions by label
        cursor.execute("SELECT label, COUNT(*) FROM predictions GROUP BY label")
        preds_by_label = dict(cursor.fetchall())
        
        # Today's predictions
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE DATE(timestamp) = DATE('now')")
        today_preds = cursor.fetchone()[0]
        
        # Active alerts
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE resolved = 0")
        active_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        # Current integrity score
        integrity_score = calculate_integrity_score(
            latest_predictions["speech"],
            latest_predictions["vision"]
        )
        risk_level, _, _ = get_risk_level(integrity_score)
        
        return {
            "status": "running",
            "models": {
                "speech_recognition": speech_model is not None,
                "computer_vision": cv_model is not None
            },
            "statistics": {
                "total_predictions": total_preds,
                "today_predictions": today_preds,
                "predictions_by_type": preds_by_type,
                "predictions_by_label": preds_by_label,
                "active_alerts": active_alerts
            },
            "current": {
                "integrity_score": integrity_score,
                "risk_level": risk_level,
                "speech_prediction": latest_predictions["speech"]["label"],
                "vision_prediction": latest_predictions["vision"]["label"]
            },
            "device": DEVICE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": {
            "speech": speech_model is not None,
            "vision": cv_model is not None
        }
    }

# =====================================================================
# ESP32 COMPATIBLE ENDPOINTS
# =====================================================================
@app.post("/esp32/audio")
async def esp32_audio_upload(
    file: UploadFile = File(...),
    device_id: str = "esp32_unknown",
    student_id: str = "unknown"
):
    """ESP32-compatible audio upload endpoint"""
    return await predict_audio_endpoint(file, device_id, student_id)

@app.post("/esp32/image")
async def esp32_image_upload(
    file: UploadFile = File(...),
    device_id: str = "esp32_unknown",
    student_id: str = "unknown"
):
    """ESP32-compatible image upload endpoint"""
    return await predict_image_endpoint(file, device_id, student_id)

# =====================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# =====================================================================
from fastapi import WebSocket
import asyncio

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send latest predictions
            data = await get_latest()
            await websocket.send_json(data)
            await asyncio.sleep(1)  # Update every second
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# =====================================================================
# START SERVER
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Argus AI Server Starting...")
    print("="*50)
    print(f"📊 Device: {DEVICE}")
    print(f"🎤 Speech Recognition: {'✅ Loaded' if speech_model else '❌ Not loaded'}")
    print(f"👁️ Computer Vision: {'✅ Loaded' if cv_model else '❌ Not loaded'}")
    print(f"📁 Database: ✅ Initialized")
    print("="*50)
    print("\n🔗 Endpoints:")
    print("  POST /predict/audio     - Predict audio")
    print("  POST /predict/image     - Predict image")
    print("  POST /predict/both      - Predict both")
    print("  GET  /latest            - Latest predictions")
    print("  GET  /history           - Prediction history")
    print("  GET  /alerts            - Recent alerts")
    print("  GET  /stats             - Server statistics")
    print("  GET  /health            - Health check")
    print("  POST /esp32/audio       - ESP32 audio upload")
    print("  POST /esp32/image       - ESP32 image upload")
    print("  WS   /ws                - WebSocket for real-time")
    print("="*50 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )