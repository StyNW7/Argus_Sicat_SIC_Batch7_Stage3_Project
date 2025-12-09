# fastapi_server_esp32.py
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import uvicorn
import time
import json
from datetime import datetime
import soundfile as sf
import io
import logging
from typing import Optional
import pandas as pd
import asyncio
from contextlib import asynccontextmanager
import redis
import sqlite3
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis (optional, for real-time updates)
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected successfully")
except:
    redis_client = None
    logger.warning("Redis not available, using in-memory storage")

# Database setup
def init_database():
    conn = sqlite3.connect('argus_data.db')
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audio_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT,
        student_id TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        audio_features TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        student_id TEXT,
        last_seen DATETIME,
        ip_address TEXT,
        status TEXT DEFAULT 'active'
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT,
        alert_type TEXT,
        severity TEXT,
        description TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        resolved BOOLEAN DEFAULT 0
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# Load ML assets
try:
    model = joblib.load("./Speech-Recognition/models_output/best_model.joblib")
    scaler = joblib.load("./Speech-Recognition/models_output/scaler.joblib")
    encoder = joblib.load("./Speech-Recognition/models_output/label_encoder.joblib")
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML models: {e}")
    raise

# Global state for latest predictions
latest_predictions = {}
connected_devices = {}
device_history = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Argus API Server...")
    
    # Load existing device data
    load_device_data()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Argus API Server...")
    save_device_data()

app = FastAPI(title="Argus AI Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(audio_data, sr=16000):
    """Extract audio features from raw audio data"""
    try:
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Extract features
        rms = np.mean(librosa.feature.rms(y=audio_np))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_np))
        spec = np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sr))
        mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        features = np.hstack([rms, zcr, spec, mfcc_mean])
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

def load_device_data():
    """Load device data from database"""
    try:
        conn = sqlite3.connect('argus_data.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT device_id, student_id, last_seen, ip_address, status FROM devices")
        devices = cursor.fetchall()
        
        for device in devices:
            device_id = device[0]
            connected_devices[device_id] = {
                'student_id': device[1],
                'last_seen': device[2],
                'ip_address': device[3],
                'status': device[4]
            }
        
        conn.close()
        logger.info(f"Loaded {len(devices)} devices from database")
    except Exception as e:
        logger.error(f"Error loading device data: {e}")

def save_device_data():
    """Save device data to database"""
    try:
        conn = sqlite3.connect('argus_data.db')
        cursor = conn.cursor()
        
        for device_id, device_data in connected_devices.items():
            cursor.execute('''
            INSERT OR REPLACE INTO devices (device_id, student_id, last_seen, ip_address, status)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                device_id,
                device_data.get('student_id'),
                device_data.get('last_seen', datetime.now().isoformat()),
                device_data.get('ip_address', ''),
                device_data.get('status', 'active')
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(connected_devices)} devices to database")
    except Exception as e:
        logger.error(f"Error saving device data: {e}")

@app.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    device_id: Optional[str] = Header(None),
    student_id: Optional[str] = Header(None),
    client_ip: Optional[str] = Header(None, alias="X-Forwarded-For")
):
    """Endpoint for ESP32 to upload audio data"""
    start_time = time.time()
    
    # Default values if headers not provided
    if not device_id:
        device_id = f"unknown_device_{int(time.time())}"
    if not student_id:
        student_id = "unknown_student"
    if not client_ip:
        client_ip = "unknown"
    
    # Update device connection
    timestamp = datetime.now().isoformat()
    connected_devices[device_id] = {
        'student_id': student_id,
        'last_seen': timestamp,
        'ip_address': client_ip,
        'status': 'active'
    }
    
    # Read audio data
    try:
        contents = await file.read()
        
        # Save raw audio for debugging (optional)
        debug_path = f"./debug_audio/{device_id}_{int(time.time())}.wav"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Received audio from device {device_id}, size: {len(contents)} bytes")
        
        # Extract features and predict
        features = extract_features(contents)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        label = encoder.inverse_transform([prediction])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))
        
        # Create prediction record
        prediction_record = {
            'device_id': device_id,
            'student_id': student_id,
            'prediction': label,
            'confidence': confidence,
            'timestamp': timestamp,
            'features': features.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        # Update latest prediction
        latest_predictions[device_id] = prediction_record
        
        # Store in database
        conn = sqlite3.connect('argus_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO audio_predictions (device_id, student_id, prediction, confidence, audio_features)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            device_id,
            student_id,
            label,
            confidence,
            json.dumps(features.tolist())
        ))
        
        # Check if this is an alert condition
        if label == 'whispering' and confidence > 0.8:
            cursor.execute('''
            INSERT INTO alerts (device_id, alert_type, severity, description)
            VALUES (?, ?, ?, ?)
            ''', (
                device_id,
                'whispering_detected',
                'medium',
                f'Whispering detected with {confidence:.2f} confidence'
            ))
        
        conn.commit()
        conn.close()
        
        # Update Redis for real-time dashboard (if available)
        if redis_client:
            redis_key = f"device:{device_id}:latest"
            redis_client.setex(redis_key, 30, json.dumps(prediction_record))
            redis_client.publish("argus:predictions", json.dumps(prediction_record))
        
        # Update device history
        if device_id not in device_history:
            device_history[device_id] = []
        
        device_history[device_id].append(prediction_record)
        if len(device_history[device_id]) > 100:  # Keep only last 100 records
            device_history[device_id] = device_history[device_id][-100:]
        
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction for device {device_id}: {label} (confidence: {confidence:.2f})")
        
        return {
            "status": "success",
            "device_id": device_id,
            "prediction": label,
            "confidence": confidence,
            "processing_time": processing_time,
            "timestamp": timestamp,
            "probabilities": {
                "silence": float(probabilities[0]),
                "whispering": float(probabilities[1]),
                "normal_conversation": float(probabilities[2])
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing audio from device {device_id}: {e}")
        
        # Log error in database
        try:
            conn = sqlite3.connect('argus_data.db')
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO alerts (device_id, alert_type, severity, description)
            VALUES (?, ?, ?, ?)
            ''', (
                device_id,
                'processing_error',
                'high',
                f'Error processing audio: {str(e)}'
            ))
            conn.commit()
            conn.close()
        except:
            pass
        
        return {
            "status": "error",
            "device_id": device_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/latest")
async def get_latest(
    device_id: Optional[str] = None,
    student_id: Optional[str] = None
):
    """Get latest predictions"""
    if device_id and device_id in latest_predictions:
        return latest_predictions[device_id]
    elif student_id:
        # Find device by student ID
        for dev_id, prediction in latest_predictions.items():
            if prediction.get('student_id') == student_id:
                return prediction
    
    # Return all latest predictions if no specific device/student requested
    return {
        "devices": latest_predictions,
        "connected_devices": len(connected_devices),
        "timestamp": time.time()
    }

@app.get("/devices")
async def get_devices():
    """Get all connected devices"""
    return {
        "devices": connected_devices,
        "count": len(connected_devices),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/device/{device_id}/history")
async def get_device_history(device_id: str, limit: int = 50):
    """Get prediction history for a specific device"""
    history = device_history.get(device_id, [])
    return {
        "device_id": device_id,
        "history": history[-limit:],
        "count": len(history)
    }

@app.get("/alerts")
async def get_alerts(limit: int = 20, resolved: bool = False):
    """Get recent alerts"""
    conn = sqlite3.connect('argus_data.db')
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
    
    alerts = cursor.fetchall()
    
    # Convert to dict
    alerts_list = []
    for alert in alerts:
        alerts_list.append({
            'id': alert[0],
            'device_id': alert[1],
            'alert_type': alert[2],
            'severity': alert[3],
            'description': alert[4],
            'timestamp': alert[5],
            'resolved': bool(alert[6])
        })
    
    conn.close()
    
    return {
        "alerts": alerts_list,
        "count": len(alerts_list),
        "unresolved_count": sum(1 for a in alerts_list if not a['resolved'])
    }

@app.post("/device/{device_id}/register")
async def register_device(
    device_id: str,
    student_id: str,
    ip_address: Optional[str] = None
):
    """Register a new device"""
    timestamp = datetime.now().isoformat()
    
    connected_devices[device_id] = {
        'student_id': student_id,
        'last_seen': timestamp,
        'ip_address': ip_address or "unknown",
        'status': 'active'
    }
    
    # Save to database
    conn = sqlite3.connect('argus_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT OR REPLACE INTO devices (device_id, student_id, last_seen, ip_address, status)
    VALUES (?, ?, ?, ?, ?)
    ''', (device_id, student_id, timestamp, ip_address or "unknown", 'active'))
    
    conn.commit()
    conn.close()
    
    return {
        "status": "success",
        "device_id": device_id,
        "student_id": student_id,
        "message": "Device registered successfully"
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    conn = sqlite3.connect('argus_data.db')
    cursor = conn.cursor()
    
    # Get total predictions
    cursor.execute("SELECT COUNT(*) FROM audio_predictions")
    total_predictions = cursor.fetchone()[0]
    
    # Get predictions by label
    cursor.execute('''
    SELECT prediction, COUNT(*) as count 
    FROM audio_predictions 
    GROUP BY prediction
    ''')
    predictions_by_label = cursor.fetchall()
    
    # Get today's predictions
    cursor.execute('''
    SELECT COUNT(*) 
    FROM audio_predictions 
    WHERE DATE(timestamp) = DATE('now')
    ''')
    today_predictions = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "server_status": "running",
        "uptime": time.time() - app_start_time,
        "connected_devices": len(connected_devices),
        "total_predictions": total_predictions,
        "today_predictions": today_predictions,
        "predictions_by_label": dict(predictions_by_label),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    device_id = None
    
    try:
        # Receive initial message with device ID
        data = await websocket.receive_text()
        message = json.loads(data)
        device_id = message.get('device_id')
        
        if device_id:
            logger.info(f"WebSocket connected for device {device_id}")
        
        while True:
            # Send latest prediction for this device
            if device_id and device_id in latest_predictions:
                await websocket.send_json(latest_predictions[device_id])
            else:
                await websocket.send_json({"status": "no_data"})
            
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"WebSocket disconnected for device {device_id}")

# Store app start time
app_start_time = time.time()

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5000,
        log_level="info"
    )