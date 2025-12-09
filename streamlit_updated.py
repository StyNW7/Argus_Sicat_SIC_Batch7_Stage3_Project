# streamlit_dashboard.py
import streamlit as st
import requests
import time
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import threading
import queue
import sounddevice as sd
import soundfile as sf
import io
import base64
import warnings
from datetime import datetime, timedelta
import asyncio
import websockets
import paho.mqtt.client as mqtt
from collections import deque
import altair as alt

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Argus - Intelligent Exam Monitoring",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #dc2626 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe-card {
        background: linear-gradient(135deg, #06b6d4 0%, #db2777 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-card {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration - UPDATED to match your FastAPI endpoints
FASTAPI_URL = "http://localhost:5000"  # Change this to your FastAPI server URL

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
    def start_recording(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
    def _record_audio(self):
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_queue.put(indata.copy())
                
        with sd.InputStream(samplerate=self.sample_rate,
                           channels=self.channels,
                           callback=callback):
            while self.is_recording:
                sd.sleep(1000)
                
    def get_audio_chunk(self, duration=3):
        """Get audio chunk of specified duration in seconds"""
        frames_needed = int(self.sample_rate * duration)
        audio_data = []
        current_frames = 0
        
        while current_frames < frames_needed and self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=1)
                audio_data.append(chunk)
                current_frames += len(chunk)
            except queue.Empty:
                break
                
        if audio_data:
            return np.concatenate(audio_data)[:frames_needed]
        return None

class VideoStreamer:
    def __init__(self):
        self.cap = None
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=10)
        
    def start_stream(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.is_streaming = True
        threading.Thread(target=self._stream_frames, daemon=True).start()
        
    def stop_stream(self):
        self.is_streaming = False
        if self.cap:
            self.cap.release()
            
    def _stream_frames(self):
        while self.is_streaming and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize for better performance
                frame = cv2.resize(frame, (640, 480))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_rgb)
            else:
                break
                
    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

def calculate_integrity_score(vision_score, audio_score):
    """Calculate integrity score based on Argus formula"""
    integrity_score = 100 - (vision_score * 0.6 + audio_score * 0.4)
    return max(0, min(100, integrity_score))

def get_risk_level(score):
    """Determine risk level based on integrity score"""
    if score >= 70:
        return "Safe", "🟢", "success"
    elif score >= 35:
        return "Alert", "🟡", "warning"
    else:
        return "Warning", "🔴", "error"

# UPDATED: Send audio to correct endpoint
def send_audio_to_api(audio_data, sample_rate=16000):
    """Send audio data to FastAPI server for prediction"""
    try:
        # Save audio to BytesIO
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
        audio_bytes.seek(0)
        
        # Send to FastAPI - UPDATED endpoint
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        response = requests.post(f"{FASTAPI_URL}/upload_audio", files=files, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "prediction": "unknown"}
            
    except Exception as e:
        return {"status": "error", "prediction": "unknown", "error": str(e)}

# UPDATED: Send frame to vision endpoint
def send_frame_to_api(frame):
    """Send frame to FastAPI server for vision prediction"""
    try:
        # Convert frame to bytes
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        img_bytes = img_encoded.tobytes()
        
        # Send to FastAPI
        files = {'file': ('frame.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(f"{FASTAPI_URL}/upload_frame", files=files, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "vision_prediction": "unknown"}
            
    except Exception as e:
        return {"status": "error", "vision_prediction": "unknown", "error": str(e)}

# UPDATED: Get latest predictions from both endpoints
def get_latest_audio_prediction():
    """Get latest audio prediction from FastAPI server"""
    try:
        response = requests.get(f"{FASTAPI_URL}/latest_audio", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Audio API Error: {str(e)}")
    return {"label": "none", "timestamp": time.time()}

def get_latest_vision_prediction():
    """Get latest vision prediction from FastAPI server"""
    try:
        response = requests.get(f"{FASTAPI_URL}/latest_vision", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Vision API Error: {str(e)}")
    return {"label": "none", "timestamp": time.time()}

def create_gauge_chart(value, title):
    """Create a gauge chart for integrity score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 35], 'color': "red"},
                {'range': [35, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_timeline_chart(history_data):
    """Create timeline chart for audio predictions"""
    if not history_data:
        return go.Figure()
        
    df = pd.DataFrame(history_data)
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Map labels to colors
    color_map = {
        'normal_conversation': 'green',
        'whispering': 'orange',
        'silence': 'blue',
        'unknown': 'gray',
        'none': 'gray'
    }
    
    fig = px.scatter(df, x='time', y='label', 
                     color='label',
                     color_discrete_map=color_map,
                     title="Audio Prediction Timeline",
                     labels={'label': 'Audio Type', 'time': 'Time'})
    
    fig.update_layout(height=300)
    return fig

def create_vision_timeline_chart(history_data):
    """Create timeline chart for vision predictions"""
    if not history_data:
        return go.Figure()
        
    df = pd.DataFrame(history_data)
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Map vision labels to colors
    color_map = {
        'focus': 'green',
        'looking_away': 'orange',
        'head_down': 'red',
        'multiple_faces': 'purple',
        'unknown': 'gray',
        'none': 'gray'
    }
    
    fig = px.scatter(df, x='time', y='label', 
                     color='label',
                     color_discrete_map=color_map,
                     title="Vision Prediction Timeline",
                     labels={'label': 'Vision Detection', 'time': 'Time'})
    
    fig.update_layout(height=300)
    return fig

def create_radar_chart(vision_metrics, audio_metrics):
    """Create radar chart for behavioral metrics"""
    categories = ['Focus', 'Gaze-off', 'Head Down', 'Face Not Detected', 
                  'Silence', 'Whispering', 'Conversation']
    
    # Normalize values for radar chart
    values = [
        vision_metrics.get('focus', 0),
        vision_metrics.get('gaze_off', 0),
        vision_metrics.get('head_down', 0),
        vision_metrics.get('face_not_detected', 0),
        audio_metrics.get('silence', 0),
        audio_metrics.get('whispering', 0),
        audio_metrics.get('conversation', 0)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) if max(values) > 0 else 10]
            )),
        showlegend=False,
        height=350
    )
    
    return fig

def main():
    # Initialize session state
    if 'audio_history' not in st.session_state:
        st.session_state.audio_history = []
    if 'vision_history' not in st.session_state:
        st.session_state.vision_history = []
    if 'integrity_scores' not in st.session_state:
        st.session_state.integrity_scores = []
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    if 'video_streamer' not in st.session_state:
        st.session_state.video_streamer = VideoStreamer()
    if 'last_audio_prediction' not in st.session_state:
        st.session_state.last_audio_prediction = {"label": "none", "timestamp": time.time()}
    if 'last_vision_prediction' not in st.session_state:
        st.session_state.last_vision_prediction = {"label": "none", "timestamp": time.time()}
    if 'vision_metrics' not in st.session_state:
        st.session_state.vision_metrics = {
            'focus': 85,
            'gaze_off': 5,
            'head_down': 3,
            'face_not_detected': 1
        }
    if 'audio_metrics' not in st.session_state:
        st.session_state.audio_metrics = {
            'silence': 70,
            'whispering': 15,
            'conversation': 15
        }
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Argus - Intelligent Exam Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Real-time AI-powered exam integrity monitoring with Computer Vision and Speech Recognition</h3>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Exam Info
        st.markdown("### 📝 Exam Information")
        exam_name = st.text_input("Exam Name", "Samsung Innovation Campus Final Exam")
        student_id = st.text_input("Student ID", "2702217125")
        student_name = st.text_input("Student Name", "Stanley Nathanael Wijaya")
        exam_duration = st.slider("Exam Duration (minutes)", 30, 180, 60)
        
        # System Controls
        st.markdown("### ⚙️ System Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Start Audio Monitoring", type="primary"):
                if not st.session_state.is_recording:
                    st.session_state.audio_recorder.start_recording()
                    st.session_state.is_recording = True
                    st.success("Audio monitoring started!")
                    
        with col2:
            if st.button("📷 Start Video Monitoring", type="primary"):
                if not st.session_state.is_streaming:
                    st.session_state.video_streamer.start_stream()
                    st.session_state.is_streaming = True
                    st.success("Video monitoring started!")
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("⏸️ Stop Audio", type="secondary"):
                if st.session_state.is_recording:
                    st.session_state.audio_recorder.stop_recording()
                    st.session_state.is_recording = False
                    st.warning("Audio monitoring stopped!")
                    
        with col4:
            if st.button("⏹️ Stop Video", type="secondary"):
                if st.session_state.is_streaming:
                    st.session_state.video_streamer.stop_stream()
                    st.session_state.is_streaming = False
                    st.warning("Video monitoring stopped!")
        
        # Threshold Settings
        st.markdown("### ⚡ Threshold Settings")
        suspicion_threshold = st.slider("Suspicion Threshold", 0, 100, 35)
        warning_threshold = st.slider("Warning Threshold", 0, 100, 70)
        
        # Connection Status
        st.markdown("### 🔗 Connection Status")
        try:
            # Test both endpoints
            audio_response = requests.get(f"{FASTAPI_URL}/latest_audio", timeout=2)
            vision_response = requests.get(f"{FASTAPI_URL}/latest_vision", timeout=2)
            
            if audio_response.status_code == 200 and vision_response.status_code == 200:
                st.success("✅ Both APIs Connected")
            elif audio_response.status_code == 200:
                st.warning("⚠️ Audio API Only")
            elif vision_response.status_code == 200:
                st.warning("⚠️ Vision API Only")
            else:
                st.error("❌ Both APIs Disconnected")
        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)}")
    
    # Main Dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Integrity Score Card
        # Calculate scores based on actual predictions
        vision_labels = st.session_state.last_vision_prediction.get('label', 'none')
        audio_labels = st.session_state.last_audio_prediction.get('label', 'none')
        
        # Calculate vision penalty
        vision_penalty = 0
        if 'looking_away' in vision_labels.lower() or 'gaze' in vision_labels.lower():
            vision_penalty += 10
        if 'head_down' in vision_labels.lower():
            vision_penalty += 15
        if 'multiple' in vision_labels.lower():
            vision_penalty += 20
        if 'no_face' in vision_labels.lower() or 'not_detected' in vision_labels.lower():
            vision_penalty += 25
            
        # Calculate audio penalty
        audio_penalty = 0
        if 'whispering' in audio_labels.lower():
            audio_penalty += 20
        if 'conversation' in audio_labels.lower():
            audio_penalty += 30
            
        integrity_score = 100 - (vision_penalty * 0.6 + audio_penalty * 0.4)
        integrity_score = max(0, min(100, integrity_score))
        
        risk_level, emoji, color = get_risk_level(integrity_score)
        
        st.markdown(f"""
        <div class="metric-card">
            <h2>Integrity Score</h2>
            <h1 style="font-size: 4rem;">{integrity_score:.1f}</h1>
            <h3>{emoji} {risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Current Status Card
        current_audio = st.session_state.last_audio_prediction.get('label', 'none').replace('_', ' ').title()
        current_vision = st.session_state.last_vision_prediction.get('label', 'none').replace('_', ' ').title()
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if "whispering" in current_audio.lower() or "looking_away" in current_vision.lower():
            card_class = "alert-card"
            status_emoji = "⚠️"
        elif integrity_score >= 70:
            card_class = "safe-card"
            status_emoji = "✅"
        else:
            card_class = "warning-card"
            status_emoji = "🔔"
            
        st.markdown(f"""
        <div class="{card_class}">
            <h2>Current Status</h2>
            <h3>{status_emoji} {risk_level}</h3>
            <p><strong>Audio:</strong> {current_audio}</p>
            <p><strong>Vision:</strong> {current_vision}</p>
            <p><strong>Time:</strong> {current_time}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Exam Info Card
        time_remaining = exam_duration * 60 - (len(st.session_state.audio_history) * 3)
        if time_remaining < 0:
            time_remaining = 0
            
        minutes = time_remaining // 60
        seconds = time_remaining % 60
        
        st.markdown(f"""
        <div class="metric-card">
            <h2>Exam Info</h2>
            <p><strong>Student:</strong> {student_name}</p>
            <p><strong>ID:</strong> {student_id}</p>
            <p><strong>Time Remaining:</strong> {minutes:02d}:{seconds:02d}</p>
            <p><strong>Alerts:</strong> {len(st.session_state.alerts)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Video and Audio Monitoring Section
    st.markdown("---")
    st.markdown("## 📊 Real-time Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Video Feed
        st.markdown("### 📷 Computer Vision Feed")
        if st.session_state.is_streaming:
            frame_placeholder = st.empty()
            label_placeholder = st.empty()
            
            # Get frame from video stream
            frame = st.session_state.video_streamer.get_frame()
            if frame is not None:
                # Display frame
                frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                
                # Send frame to FastAPI for vision prediction
                with st.spinner("Analyzing video..."):
                    prediction = send_frame_to_api(frame)
                    
                    if prediction.get('status') == 'ok':
                        label = prediction.get('vision_prediction', 'unknown')
                        timestamp = time.time()
                        
                        # Update last prediction
                        st.session_state.last_vision_prediction = {
                            'label': label,
                            'timestamp': timestamp
                        }
                        
                        # Add to history
                        st.session_state.vision_history.append({
                            'label': label,
                            'timestamp': timestamp
                        })
                        
                        # Keep only last 50 entries
                        if len(st.session_state.vision_history) > 50:
                            st.session_state.vision_history = st.session_state.vision_history[-50:]
                        
                        # Update vision metrics based on prediction
                        label_lower = label.lower()
                        if 'focus' in label_lower or 'normal' in label_lower:
                            st.session_state.vision_metrics['focus'] += 1
                        elif 'looking' in label_lower or 'gaze' in label_lower:
                            st.session_state.vision_metrics['gaze_off'] += 1
                            # Add alert for suspicious activity
                            alert_msg = f"Vision: Looking away detected at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}"
                            if alert_msg not in st.session_state.alerts:
                                st.session_state.alerts.append(alert_msg)
                        elif 'head' in label_lower:
                            st.session_state.vision_metrics['head_down'] += 1
                            alert_msg = f"Vision: Head down detected at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}"
                            if alert_msg not in st.session_state.alerts:
                                st.session_state.alerts.append(alert_msg)
                        elif 'multiple' in label_lower or 'face' in label_lower:
                            st.session_state.vision_metrics['face_not_detected'] += 1
                            alert_msg = f"Vision: Multiple faces detected at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}"
                            if alert_msg not in st.session_state.alerts:
                                st.session_state.alerts.append(alert_msg)
                
                # Display prediction
                label_display = label.replace('_', ' ').title()
                if 'focus' in label.lower() or 'normal' in label.lower():
                    label_color = 'green'
                elif 'looking' in label.lower() or 'head' in label.lower():
                    label_color = 'orange'
                else:
                    label_color = 'red'
                    
                label_placeholder.markdown(f"""
                <div style="background-color: {label_color};
                            padding: 10px; border-radius: 5px; text-align: center; margin-top: 10px;">
                    <h3 style="color: white; margin: 0;">
                        {label_display}
                    </h3>
                    <p style="color: white; margin: 0;">Updated: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click 'Start Video Monitoring' to begin video feed")
    
    with col2:
        # Audio Monitoring
        st.markdown("### 🎤 Speech Recognition")
        
        if st.session_state.is_recording:
            # Get audio chunk and send to API
            audio_chunk = st.session_state.audio_recorder.get_audio_chunk(duration=3)
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Send to FastAPI server
                with st.spinner("Analyzing audio..."):
                    prediction = send_audio_to_api(audio_chunk)
                    
                    if prediction.get('status') == 'ok':
                        label = prediction.get('prediction', 'unknown')
                        timestamp = time.time()
                        
                        # Update last prediction
                        st.session_state.last_audio_prediction = {
                            'label': label,
                            'timestamp': timestamp
                        }
                        
                        # Add to history
                        st.session_state.audio_history.append({
                            'label': label,
                            'timestamp': timestamp
                        })
                        
                        # Keep only last 50 entries
                        if len(st.session_state.audio_history) > 50:
                            st.session_state.audio_history = st.session_state.audio_history[-50:]
                        
                        # Update audio metrics
                        label_lower = label.lower()
                        if 'silence' in label_lower:
                            st.session_state.audio_metrics['silence'] += 5
                        elif 'whispering' in label_lower:
                            st.session_state.audio_metrics['whispering'] += 10
                            # Add alert
                            alert_msg = f"Audio: Whispering detected at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}"
                            if alert_msg not in st.session_state.alerts:
                                st.session_state.alerts.append(alert_msg)
                        elif 'conversation' in label_lower:
                            st.session_state.audio_metrics['conversation'] += 10
                            alert_msg = f"Audio: Conversation detected at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}"
                            if alert_msg not in st.session_state.alerts:
                                st.session_state.alerts.append(alert_msg)
                
                # Display current audio prediction
                label_display = label.replace('_', ' ').title()
                color_map = {
                    'silence': 'blue',
                    'whispering': 'orange',
                    'normal_conversation': 'red',
                    'unknown': 'gray'
                }
                color = color_map.get(label, 'gray')
                
                st.markdown(f"""
                <div style="background-color: {color};
                            padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
                    <h2 style="color: white; margin: 0;">Current Audio: {label_display}</h2>
                    <p style="color: white; margin: 0;">Last updated: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Audio waveform visualization
                st.markdown("**Audio Waveform**")
                if len(audio_chunk) > 0:
                    chart_data = pd.DataFrame({
                        'Time': np.arange(len(audio_chunk)) / 16000,
                        'Amplitude': audio_chunk.flatten()
                    })
                    
                    line_chart = alt.Chart(chart_data.iloc[::100]).mark_line().encode(
                        x=alt.X('Time:Q', title='Time (s)'),
                        y=alt.Y('Amplitude:Q', title='Amplitude'),
                        color=alt.value(color)
                    ).properties(height=200)
                    
                    st.altair_chart(line_chart, use_container_width=True)
            else:
                st.info("Waiting for audio data...")
        else:
            st.info("Click 'Start Audio Monitoring' to begin audio analysis")
    
    # Charts and Analytics Section
    st.markdown("---")
    st.markdown("## 📈 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge Chart for Integrity Score
        fig_gauge = create_gauge_chart(integrity_score, "Integrity Score")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Behavioral Radar Chart
        st.markdown("### Behavioral Metrics")
        fig_radar = create_radar_chart(
            st.session_state.vision_metrics,
            st.session_state.audio_metrics
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Timeline Charts in tabs
        tab1, tab2 = st.tabs(["Audio Timeline", "Vision Timeline"])
        
        with tab1:
            st.markdown("### Audio Prediction Timeline")
            if st.session_state.audio_history:
                fig_timeline = create_timeline_chart(st.session_state.audio_history)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No audio data yet. Start audio monitoring to see timeline.")
        
        with tab2:
            st.markdown("### Vision Prediction Timeline")
            if st.session_state.vision_history:
                fig_timeline = create_vision_timeline_chart(st.session_state.vision_history)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No vision data yet. Start video monitoring to see timeline.")
        
        # Metrics Distribution
        st.markdown("### Real-time Metrics")
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.metric("Focus Time", f"{st.session_state.vision_metrics.get('focus', 0)}%")
            st.metric("Gaze-off", f"{st.session_state.vision_metrics.get('gaze_off', 0)} events")
            st.metric("Head Down", f"{st.session_state.vision_metrics.get('head_down', 0)} events")
            
        with col_metrics2:
            st.metric("Silence", f"{st.session_state.audio_metrics.get('silence', 0)}%")
            st.metric("Whispering", f"{st.session_state.audio_metrics.get('whispering', 0)} events")
            st.metric("Conversation", f"{st.session_state.audio_metrics.get('conversation', 0)}%")
    
    # Alerts and History Section
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Recent Alerts")
        if st.session_state.alerts:
            for i, alert in enumerate(reversed(st.session_state.alerts[-5:])):
                st.warning(f"{alert}")
        else:
            st.info("No alerts detected")
            
        if st.button("Clear Alerts"):
            st.session_state.alerts = []
            st.rerun()
    
    with col2:
        st.markdown("### 📋 Recent Predictions")
        tab_audio, tab_vision = st.tabs(["Audio", "Vision"])
        
        with tab_audio:
            if st.session_state.audio_history:
                recent_data = st.session_state.audio_history[-5:]
                for data in reversed(recent_data):
                    label = data['label'].replace('_', ' ').title()
                    time_str = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')
                    st.write(f"**{time_str}**: {label}")
            else:
                st.info("No audio predictions yet")
                
        with tab_vision:
            if st.session_state.vision_history:
                recent_data = st.session_state.vision_history[-5:]
                for data in reversed(recent_data):
                    label = data['label'].replace('_', ' ').title()
                    time_str = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')
                    st.write(f"**{time_str}**: {label}")
            else:
                st.info("No vision predictions yet")
    
    # Export and Report Section
    st.markdown("---")
    st.markdown("## 📊 Report Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Report as CSV"):
            # Create report data
            report_data = {
                'Student Name': [student_name],
                'Student ID': [student_id],
                'Exam Name': [exam_name],
                'Final Integrity Score': [integrity_score],
                'Risk Level': [risk_level],
                'Total Alerts': [len(st.session_state.alerts)],
                'Whispering Events': [st.session_state.audio_metrics.get('whispering', 0)],
                'Conversation Events': [st.session_state.audio_metrics.get('conversation', 0)],
                'Gaze-off Events': [st.session_state.vision_metrics.get('gaze_off', 0)],
                'Head Down Events': [st.session_state.vision_metrics.get('head_down', 0)],
                'Face Not Detected': [st.session_state.vision_metrics.get('face_not_detected', 0)],
                'Last Audio Prediction': [current_audio],
                'Last Vision Prediction': [current_vision],
                'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            
            df_report = pd.DataFrame(report_data)
            csv = df_report.to_csv(index=False)
            
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"argus_report_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Reset Monitoring"):
            st.session_state.audio_history = []
            st.session_state.vision_history = []
            st.session_state.integrity_scores = []
            st.session_state.alerts = []
            st.session_state.last_audio_prediction = {"label": "none", "timestamp": time.time()}
            st.session_state.last_vision_prediction = {"label": "none", "timestamp": time.time()}
            st.session_state.vision_metrics = {
                'focus': 0,
                'gaze_off': 0,
                'head_down': 0,
                'face_not_detected': 0
            }
            st.session_state.audio_metrics = {
                'silence': 0,
                'whispering': 0,
                'conversation': 0
            }
            st.rerun()
    
    with col3:
        if st.button("📊 View Detailed Analytics"):
            st.markdown("### Detailed Analytics")
            analytics_df = pd.DataFrame({
                'Metric': list(st.session_state.vision_metrics.keys()) + list(st.session_state.audio_metrics.keys()),
                'Value': list(st.session_state.vision_metrics.values()) + list(st.session_state.audio_metrics.values()),
                'Type': ['Vision'] * len(st.session_state.vision_metrics) + ['Audio'] * len(st.session_state.audio_metrics)
            })
            
            st.dataframe(analytics_df)
            
            # Create bar chart
            fig_bar = px.bar(analytics_df, x='Metric', y='Value', color='Type',
                            title="Detailed Behavioral Metrics")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p><strong>Argus - Integrity Through Intelligent Vision</strong></p>
        <p>Samsung Innovation Campus Batch 7 Stage 3 | Team Sicat</p>
        <p>© 2025 BINUS UNIVERSITY. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh for real-time updates
    if st.session_state.is_recording or st.session_state.is_streaming:
        time.sleep(3)  # Update every 3 seconds
        st.rerun()

if __name__ == "__main__":
    main()