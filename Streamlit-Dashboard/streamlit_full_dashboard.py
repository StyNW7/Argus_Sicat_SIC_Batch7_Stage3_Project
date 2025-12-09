# streamlit_dashboard_complete.py
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
from PIL import Image
import io
import base64
import warnings
from datetime import datetime, timedelta
import asyncio
import websockets
import sqlite3
from collections import deque
import altair as alt
import threading
import queue
import sounddevice as sd
import soundfile as sf
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================
FASTAPI_URL = "http://localhost:5000"  # Change to your server URL
DATABASE_PATH = "argus_dashboard.db"
REFRESH_INTERVAL = 5  # seconds for auto-refresh

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.set_page_config(
    page_title="Argus - AI Exam Monitoring Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Sub header */
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #6B7280;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-safe {
        background-color: #10B98120;
        color: #10B981;
        border: 1px solid #10B98140;
    }
    
    .status-alert {
        background-color: #F59E0B20;
        color: #F59E0B;
        border: 1px solid #F59E0B40;
    }
    
    .status-warning {
        background-color: #EF444420;
        color: #EF4444;
        border: 1px solid #EF444440;
    }
    
    /* Device cards */
    .device-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem;
        border: 1px solid #E5E7EB;
        transition: all 0.2s ease;
    }
    
    .device-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
    
    .device-card.active {
        border: 2px solid #3B82F6;
        background-color: #EFF6FF;
    }
    
    /* Alert cards */
    .alert-card {
        background: linear-gradient(135deg, #FEE2E2, #FEF3C7);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid;
        margin-bottom: 0.75rem;
    }
    
    .alert-high {
        border-left-color: #DC2626;
    }
    
    .alert-medium {
        border-left-color: #F59E0B;
    }
    
    .alert-low {
        border-left-color: #3B82F6;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 10px 10px 0 0;
        padding: 0 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# API CLIENT CLASS
# =====================================================================
class ArgusAPIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10
        
    def test_connection(self):
        """Test connection to FastAPI server"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_server_stats(self):
        """Get server statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error getting stats: {e}")
        return {}
    
    def get_latest_predictions(self):
        """Get latest predictions"""
        try:
            response = self.session.get(f"{self.base_url}/latest", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error getting latest predictions: {e}")
        return {}
    
    def get_history(self, limit=50):
        """Get prediction history"""
        try:
            response = self.session.get(f"{self.base_url}/history?limit={limit}", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error getting history: {e}")
        return {}
    
    def get_alerts(self, limit=20, resolved=False):
        """Get alerts"""
        try:
            resolved_str = "true" if resolved else "false"
            response = self.session.get(f"{self.base_url}/alerts?limit={limit}&resolved={resolved_str}", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error getting alerts: {e}")
        return {}
    
    def predict_audio(self, audio_bytes, device_id="dashboard", student_id="unknown"):
        """Send audio for prediction"""
        try:
            files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
            data = {
                'device_id': device_id,
                'student_id': student_id
            }
            response = self.session.post(
                f"{self.base_url}/predict/audio",
                files=files,
                data=data,
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error predicting audio: {e}")
        return {}
    
    def predict_image(self, image_bytes, device_id="dashboard", student_id="unknown"):
        """Send image for prediction"""
        try:
            files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
            data = {
                'device_id': device_id,
                'student_id': student_id
            }
            response = self.session.post(
                f"{self.base_url}/predict/image",
                files=files,
                data=data,
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error predicting image: {e}")
        return {}
    
    def predict_both(self, audio_bytes=None, image_bytes=None, device_id="dashboard", student_id="unknown"):
        """Send both audio and image"""
        try:
            files = {}
            if audio_bytes:
                files['audio_file'] = ('audio.wav', audio_bytes, 'audio/wav')
            if image_bytes:
                files['image_file'] = ('image.jpg', image_bytes, 'image/jpeg')
            
            data = {
                'device_id': device_id,
                'student_id': student_id
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/both",
                files=files,
                data=data,
                timeout=20
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error predicting both: {e}")
        return {}

# =====================================================================
# DASHBOARD COMPONENTS
# =====================================================================
class DashboardComponents:
    @staticmethod
    def create_integrity_gauge(score, title="Integrity Score"):
        """Create gauge chart for integrity score"""
        risk_color = {
            "Safe": "#10B981",
            "Alert": "#F59E0B",
            "Warning": "#EF4444"
        }.get("Safe" if score >= 70 else "Alert" if score >= 35 else "Warning", "#3B82F6")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={'text': title, 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "#10B981"}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': risk_color, 'thickness': 0.4},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 35], 'color': '#FEE2E2'},
                    {'range': [35, 70], 'color': '#FEF3C7'},
                    {'range': [70, 100], 'color': '#D1FAE5'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
    
    @staticmethod
    def create_prediction_timeline(history_data):
        """Create timeline chart for predictions"""
        if not history_data or 'history' not in history_data:
            return go.Figure()
        
        df = pd.DataFrame(history_data['history'])
        if df.empty:
            return go.Figure()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            return go.Figure()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_minute'] = df['timestamp'].dt.strftime('%H:%M')
        
        # Group by time and prediction type
        if 'prediction_type' in df.columns and 'label' in df.columns:
            timeline_data = df.groupby(['hour_minute', 'prediction_type', 'label']).size().reset_index(name='count')
            
            color_map = {
                'speech': {'silence': '#3B82F6', 'whispering': '#EF4444', 'normal_conversation': '#10B981'},
                'vision': {'focus': '#10B981', 'looking_away': '#F59E0B', 'cheating': '#EF4444'}
            }
            
            fig = px.scatter(timeline_data, x='hour_minute', y='prediction_type',
                           size='count', color='label',
                           title="Prediction Timeline",
                           labels={'hour_minute': 'Time', 'prediction_type': 'Type', 'label': 'Prediction'})
            
            fig.update_layout(
                height=350,
                xaxis_title="Time",
                yaxis_title="Prediction Type",
                showlegend=True
            )
            
            return fig
        
        return go.Figure()
    
    @staticmethod
    def create_distribution_chart(prediction_data):
        """Create distribution chart for predictions"""
        if not prediction_data:
            return go.Figure()
        
        labels = []
        counts = []
        colors = []
        
        # Process speech predictions
        speech_pred = prediction_data.get('speech', {}).get('label', 'none')
        if speech_pred != 'none':
            labels.append(f"Speech: {speech_pred}")
            counts.append(1)
            colors.append('#3B82F6' if 'silence' in speech_pred else '#EF4444' if 'whispering' in speech_pred else '#10B981')
        
        # Process vision predictions
        vision_pred = prediction_data.get('vision', {}).get('label', 'none')
        if vision_pred != 'none':
            labels.append(f"Vision: {vision_pred}")
            counts.append(1)
            colors.append('#10B981' if 'focus' in vision_pred else '#F59E0B' if 'away' in vision_pred else '#EF4444')
        
        if not labels:
            return go.Figure()
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=counts,
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent',
            hoverinfo='label+value'
        )])
        
        fig.update_layout(
            title="Current Predictions Distribution",
            height=300,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_confidence_meter(confidence, title="Confidence"):
        """Create confidence meter"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': title},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#3B82F6"},
                'steps': [
                    {'range': [0, 60], 'color': "#FEE2E2"},
                    {'range': [60, 80], 'color': "#FEF3C7"},
                    {'range': [80, 100], 'color': "#D1FAE5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence * 100
                }
            }
        ))
        
        fig.update_layout(height=250)
        return fig

# =====================================================================
# CAMERA CAPTURE COMPONENT
# =====================================================================
class CameraCapture:
    def __init__(self):
        self.cap = None
        self.is_capturing = False
        self.frame_queue = queue.Queue(maxsize=1)
        
    def start_capture(self, camera_index=0):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                return False
            
            self.is_capturing = True
            threading.Thread(target=self._capture_frames, daemon=True).start()
            return True
        except Exception as e:
            st.error(f"Error starting camera: {e}")
            return False
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_capturing = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _capture_frames(self):
        """Capture frames in background thread"""
        while self.is_capturing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize for performance
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame_rgb)
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame(self):
        """Get latest frame"""
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

# =====================================================================
# AUDIO RECORDER COMPONENT
# =====================================================================
class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_data = None
        self.recording_thread = None
        
    def start_recording(self, duration=3):
        """Record audio for specified duration"""
        try:
            self.audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            self.is_recording = True
            
            # Wait for recording to complete
            sd.wait()
            self.is_recording = False
            
            return True
        except Exception as e:
            st.error(f"Error recording audio: {e}")
            return False
    
    def get_audio_bytes(self):
        """Convert audio data to WAV bytes"""
        if self.audio_data is not None:
            buffer = io.BytesIO()
            sf.write(buffer, self.audio_data, self.sample_rate, format='WAV')
            buffer.seek(0)
            return buffer.getvalue()
        return None

# =====================================================================
# MAIN DASHBOARD
# =====================================================================
def main():
    # Initialize session state
    if 'api_client' not in st.session_state:
        st.session_state.api_client = ArgusAPIClient(FASTAPI_URL)
    
    if 'camera' not in st.session_state:
        st.session_state.camera = CameraCapture()
    
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    
    if 'latest_data' not in st.session_state:
        st.session_state.latest_data = {}
    
    if 'history_data' not in st.session_state:
        st.session_state.history_data = {}
    
    if 'alerts_data' not in st.session_state:
        st.session_state.alerts_data = {}
    
    if 'is_camera_on' not in st.session_state:
        st.session_state.is_camera_on = False
    
    if 'is_audio_on' not in st.session_state:
        st.session_state.is_audio_on = False
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'selected_student' not in st.session_state:
        st.session_state.selected_student = "2702217125"  # Default student
    
    if 'device_id' not in st.session_state:
        st.session_state.device_id = "dashboard_pc"
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Argus AI Exam Monitoring Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time monitoring of exam integrity using AI-powered Computer Vision and Speech Recognition</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        
        # Server Configuration
        st.markdown("### 🔌 Server Connection")
        server_url = st.text_input("FastAPI Server URL", FASTAPI_URL)
        if server_url != FASTAPI_URL:
            st.session_state.api_client.base_url = server_url
        
        # Test connection
        if st.button("🔗 Test Connection", type="primary"):
            if st.session_state.api_client.test_connection():
                st.success("✅ Connected to server!")
            else:
                st.error("❌ Cannot connect to server")
        
        # Exam Configuration
        st.markdown("### 📝 Exam Configuration")
        exam_name = st.text_input("Exam Name", "Samsung Innovation Campus Final Exam")
        exam_duration = st.slider("Duration (minutes)", 30, 180, 90)
        
        # Student Selection
        students = {
            "2702217125": "Stanley Nathanael Wijaya",
            "2702312865": "Clarissa Aditjakra", 
            "2702254543": "Jazzlyn Amelia Lim",
            "2702220611": "Visella"
        }
        
        selected_student_id = st.selectbox(
            "Select Student",
            list(students.keys()),
            format_func=lambda x: f"{x} - {students[x]}"
        )
        st.session_state.selected_student = selected_student_id
        
        # Device Configuration
        st.markdown("### 📱 Device Settings")
        st.session_state.device_id = st.text_input("Device ID", "dashboard_pc")
        
        # Auto-refresh Settings
        st.markdown("### 🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto-refresh", value=True)
        refresh_interval = st.slider("Interval (seconds)", 2, 30, REFRESH_INTERVAL)
        st.session_state.auto_refresh = auto_refresh
        
        if st.button("🔄 Manual Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Connection Status
        st.markdown("### 📊 Connection Status")
        try:
            stats = st.session_state.api_client.get_server_stats()
            if stats:
                st.success("✅ Server Connected")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Uptime", f"{stats.get('current', {}).get('integrity_score', 0):.0f}s")
                with col2:
                    st.metric("Predictions", stats.get('statistics', {}).get('total_predictions', 0))
            else:
                st.error("❌ Server Offline")
        except:
            st.error("❌ Connection Error")
    
    # Main Dashboard Content
    api_client = st.session_state.api_client
    
    # Fetch latest data
    latest_data = api_client.get_latest_predictions()
    history_data = api_client.get_history(limit=30)
    alerts_data = api_client.get_alerts(limit=10)
    stats_data = api_client.get_server_stats()
    
    # Update session state
    st.session_state.latest_data = latest_data
    st.session_state.history_data = history_data
    st.session_state.alerts_data = alerts_data
    
    # Calculate metrics
    integrity_score = latest_data.get('integrity_score', 100)
    risk_level = latest_data.get('risk_level', 'Safe')
    risk_emoji = latest_data.get('risk_emoji', '🟢')
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3B82F6;">
            <p class="metric-value">{integrity_score:.1f}</p>
            <p class="metric-label">Integrity Score</p>
            <span class="status-badge status-{risk_level.lower()}">{risk_emoji} {risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        speech_pred = latest_data.get('speech', {}).get('label', 'none')
        speech_conf = latest_data.get('speech', {}).get('confidence', 0)
        speech_display = speech_pred.replace('_', ' ').title() if speech_pred != 'none' else 'No Data'
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #10B981;">
            <p class="metric-value">{speech_conf*100:.1f}%</p>
            <p class="metric-label">Speech: {speech_display}</p>
            <small>Last updated</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vision_pred = latest_data.get('vision', {}).get('label', 'none')
        vision_conf = latest_data.get('vision', {}).get('confidence', 0)
        vision_display = vision_pred.replace('_', ' ').title() if vision_pred != 'none' else 'No Data'
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #EF4444;">
            <p class="metric-value">{vision_conf*100:.1f}%</p>
            <p class="metric-label">Vision: {vision_display}</p>
            <small>Last updated</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        active_alerts = alerts_data.get('count', 0)
        alert_color = "#EF4444" if active_alerts > 0 else "#10B981"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {alert_color};">
            <p class="metric-value">{active_alerts}</p>
            <p class="metric-label">Active Alerts</p>
            <small>Require attention</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Monitoring", 
        "🎥 Computer Vision", 
        "🎤 Speech Recognition", 
        "📈 Analytics", 
        "⚠️ Alerts"
    ])
    
    with tab1:
        # Live Monitoring Dashboard
        st.markdown("### 📡 Real-time Monitoring")
        
        # Create two columns for charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Integrity Score Gauge
            fig_gauge = DashboardComponents.create_integrity_gauge(integrity_score)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            col_action1, col_action2 = st.columns(2)
            
            with col_action1:
                if st.button("📸 Capture & Analyze Image", type="secondary", use_container_width=True):
                    if not st.session_state.is_camera_on:
                        if st.session_state.camera.start_capture():
                            st.session_state.is_camera_on = True
                            st.success("Camera started!")
                        else:
                            st.error("Failed to start camera")
                    else:
                        frame = st.session_state.camera.get_frame()
                        if frame is not None:
                            # Convert frame to bytes
                            img_pil = Image.fromarray(frame)
                            img_bytes = io.BytesIO()
                            img_pil.save(img_bytes, format='JPEG')
                            img_bytes = img_bytes.getvalue()
                            
                            # Send for prediction
                            with st.spinner("Analyzing image..."):
                                result = api_client.predict_image(
                                    img_bytes,
                                    st.session_state.device_id,
                                    st.session_state.selected_student
                                )
                            
                            if result:
                                st.success(f"Vision Prediction: {result.get('prediction', 'Unknown')}")
            
            with col_action2:
                if st.button("🎤 Record & Analyze Audio", type="secondary", use_container_width=True):
                    with st.spinner("Recording audio (3 seconds)..."):
                        if st.session_state.audio_recorder.start_recording(duration=3):
                            audio_bytes = st.session_state.audio_recorder.get_audio_bytes()
                            if audio_bytes:
                                result = api_client.predict_audio(
                                    audio_bytes,
                                    st.session_state.device_id,
                                    st.session_state.selected_student
                                )
                                if result:
                                    st.success(f"Audio Prediction: {result.get('prediction', 'Unknown')}")
        
        with col_chart2:
            # Prediction Distribution
            fig_dist = DashboardComponents.create_distribution_chart(latest_data)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Current Predictions Details
            st.markdown("### 🔍 Current Predictions")
            
            # Speech Prediction
            with st.expander("🎤 Speech Recognition Details", expanded=True):
                speech_data = latest_data.get('speech', {})
                if speech_data.get('label', 'none') != 'none':
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.metric("Label", speech_data['label'].replace('_', ' ').title())
                    with col_s2:
                        st.metric("Confidence", f"{speech_data.get('confidence', 0)*100:.1f}%")
                else:
                    st.info("No recent speech prediction")
            
            # Vision Prediction
            with st.expander("👁️ Computer Vision Details", expanded=True):
                vision_data = latest_data.get('vision', {})
                if vision_data.get('label', 'none') != 'none':
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.metric("Label", vision_data['label'].replace('_', ' ').title())
                    with col_v2:
                        st.metric("Confidence", f"{vision_data.get('confidence', 0)*100:.1f}%")
                    
                    # Show probabilities if available
                    if 'probabilities' in vision_data:
                        probs = vision_data['probabilities']
                        if probs and len(probs) > 0:
                            st.markdown("**Class Probabilities:**")
                            for i, prob in enumerate(probs[:5]):  # Show top 5
                                st.progress(float(prob), text=f"Class {i}: {prob:.2%}")
                else:
                    st.info("No recent vision prediction")
    
    with tab2:
        # Computer Vision Tab
        st.markdown("### 🎥 Computer Vision Monitoring")
        
        col_cam1, col_cam2 = st.columns([3, 2])
        
        with col_cam1:
            # Camera Feed
            st.markdown("#### Live Camera Feed")
            
            camera_placeholder = st.empty()
            controls_placeholder = st.container()
            
            with controls_placeholder:
                col_cam_ctrl1, col_cam_ctrl2 = st.columns(2)
                with col_cam_ctrl1:
                    if st.button("▶️ Start Camera", type="primary", use_container_width=True):
                        if not st.session_state.is_camera_on:
                            if st.session_state.camera.start_capture():
                                st.session_state.is_camera_on = True
                                st.success("Camera started!")
                                st.rerun()
                
                with col_cam_ctrl2:
                    if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True):
                        if st.session_state.is_camera_on:
                            st.session_state.camera.stop_capture()
                            st.session_state.is_camera_on = False
                            st.warning("Camera stopped!")
                            st.rerun()
            
            # Display camera feed
            if st.session_state.is_camera_on:
                frame = st.session_state.camera.get_frame()
                if frame is not None:
                    camera_placeholder.image(frame, channels="RGB", use_column_width=True)
                    
                    # Analyze button
                    if st.button("🔍 Analyze Current Frame", type="primary"):
                        # Convert frame to bytes
                        img_pil = Image.fromarray(frame)
                        img_bytes = io.BytesIO()
                        img_pil.save(img_bytes, format='JPEG')
                        img_bytes = img_bytes.getvalue()
                        
                        # Send for prediction
                        with st.spinner("Analyzing image with AI..."):
                            result = api_client.predict_image(
                                img_bytes,
                                st.session_state.device_id,
                                st.session_state.selected_student
                            )
                        
                        if result:
                            st.success(f"✅ Prediction: {result.get('prediction', 'Unknown')}")
                            st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
            else:
                camera_placeholder.info("Camera is off. Click 'Start Camera' to begin.")
        
        with col_cam2:
            # Vision Analytics
            st.markdown("#### Vision Analytics")
            
            # Confidence meter
            vision_conf = latest_data.get('vision', {}).get('confidence', 0)
            fig_confidence = DashboardComponents.create_confidence_meter(vision_conf)
            st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Recent vision predictions
            st.markdown("#### Recent Vision Predictions")
            if history_data and 'history' in history_data:
                vision_history = [h for h in history_data['history'] if h.get('prediction_type') == 'vision']
                if vision_history:
                    df_vision = pd.DataFrame(vision_history[-5:])
                    st.dataframe(df_vision[['timestamp', 'label', 'confidence', 'integrity_score']], use_container_width=True)
                else:
                    st.info("No vision predictions yet")
            
            # Upload image for analysis
            st.markdown("#### Upload Image for Analysis")
            uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_image is not None:
                # Display image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Analyze button
                if st.button("🤖 Analyze Uploaded Image", type="primary"):
                    img_bytes = uploaded_image.getvalue()
                    
                    with st.spinner("Analyzing image..."):
                        result = api_client.predict_image(
                            img_bytes,
                            st.session_state.device_id,
                            st.session_state.selected_student
                        )
                    
                    if result:
                        st.success(f"✅ Prediction: {result.get('prediction', 'Unknown')}")
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.metric("Label", result.get('prediction', 'Unknown'))
                        with col_res2:
                            st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
    
    with tab3:
        # Speech Recognition Tab
        st.markdown("### 🎤 Speech Recognition Monitoring")
        
        col_audio1, col_audio2 = st.columns([3, 2])
        
        with col_audio1:
            # Audio Recording Controls
            st.markdown("#### Audio Recording")
            
            # Record duration
            record_duration = st.slider("Recording Duration (seconds)", 1, 10, 3)
            
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                    with st.spinner(f"Recording audio for {record_duration} seconds..."):
                        if st.session_state.audio_recorder.start_recording(record_duration):
                            st.success("Recording complete!")
                            
                            # Get audio bytes and analyze
                            audio_bytes = st.session_state.audio_recorder.get_audio_bytes()
                            if audio_bytes:
                                result = api_client.predict_audio(
                                    audio_bytes,
                                    st.session_state.device_id,
                                    st.session_state.selected_student
                                )
                                
                                if result:
                                    st.success(f"✅ Prediction: {result.get('prediction', 'Unknown')}")
                                    col_ar1, col_ar2 = st.columns(2)
                                    with col_ar1:
                                        st.metric("Label", result.get('prediction', 'Unknown'))
                                    with col_ar2:
                                        st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
            
            with col_rec2:
                # Play last recording button
                if st.button("🔊 Play Last Recording", type="secondary", use_container_width=True):
                    audio_bytes = st.session_state.audio_recorder.get_audio_bytes()
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/wav')
                    else:
                        st.warning("No audio recorded yet")
            
            # Upload audio for analysis
            st.markdown("#### Upload Audio for Analysis")
            uploaded_audio = st.file_uploader("Choose an audio file...", type=['wav', 'mp3', 'm4a'])
            
            if uploaded_audio is not None:
                # Display audio player
                st.audio(uploaded_audio, format='audio/wav')
                
                # Analyze button
                if st.button("🤖 Analyze Uploaded Audio", type="primary"):
                    audio_bytes = uploaded_audio.getvalue()
                    
                    with st.spinner("Analyzing audio..."):
                        result = api_client.predict_audio(
                            audio_bytes,
                            st.session_state.device_id,
                            st.session_state.selected_student
                        )
                    
                    if result:
                        st.success(f"✅ Prediction: {result.get('prediction', 'Unknown')}")
                        col_ua1, col_ua2 = st.columns(2)
                        with col_ua1:
                            st.metric("Label", result.get('prediction', 'Unknown'))
                        with col_ua2:
                            st.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
            
            # Speech prediction history
            st.markdown("#### Speech Prediction History")
            if history_data and 'history' in history_data:
                speech_history = [h for h in history_data['history'] if h.get('prediction_type') == 'speech']
                if speech_history:
                    df_speech = pd.DataFrame(speech_history[-10:])
                    st.dataframe(df_speech[['timestamp', 'label', 'confidence', 'integrity_score']], use_container_width=True)
                else:
                    st.info("No speech predictions yet")
        
        with col_audio2:
            # Audio Analytics
            st.markdown("#### Audio Analytics")
            
            # Confidence meter for speech
            speech_conf = latest_data.get('speech', {}).get('confidence', 0)
            fig_speech_confidence = DashboardComponents.create_confidence_meter(speech_conf, "Speech Confidence")
            st.plotly_chart(fig_speech_confidence, use_container_width=True)
            
            # Audio features visualization (simulated)
            st.markdown("#### Audio Features")
            
            # Create simulated audio features
            features = {
                'RMS Energy': np.random.uniform(0.1, 0.5),
                'Zero Crossing Rate': np.random.uniform(0.01, 0.2),
                'Spectral Centroid': np.random.uniform(1000, 5000),
                'MFCC 1': np.random.uniform(-600, -200),
                'MFCC 2': np.random.uniform(50, 150)
            }
            
            df_features = pd.DataFrame({
                'Feature': list(features.keys()),
                'Value': list(features.values())
            })
            
            fig_features = px.bar(df_features, x='Feature', y='Value', title="Audio Feature Values")
            fig_features.update_layout(height=300)
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Real-time audio monitoring status
            st.markdown("#### Monitoring Status")
            col_status1, col_status2 = st.columns(2)
            with col_status1:
                st.metric("Sample Rate", "16 kHz")
            with col_status2:
                st.metric("Channels", "Mono")
    
    with tab4:
        # Analytics Tab
        st.markdown("### 📈 Advanced Analytics")
        
        # Create three columns for different charts
        col_anal1, col_anal2 = st.columns(2)
        
        with col_anal1:
            # Prediction Timeline
            st.markdown("#### Prediction Timeline")
            fig_timeline = DashboardComponents.create_prediction_timeline(history_data)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No prediction history available yet")
            
            # Integrity Score History
            st.markdown("#### Integrity Score History")
            if integrity_score_history := st.session_state.get('integrity_score_history', []):
                df_scores = pd.DataFrame(integrity_score_history)
                fig_scores = px.line(df_scores, x='timestamp', y='score', 
                                    title="Integrity Score Over Time")
                fig_scores.update_layout(height=300)
                st.plotly_chart(fig_scores, use_container_width=True)
            else:
                # Simulate some data for demo
                times = pd.date_range(end=datetime.now(), periods=20, freq='1min')
                scores = np.random.normal(70, 15, 20).clip(0, 100)
                df_scores = pd.DataFrame({'timestamp': times, 'score': scores})
                fig_scores = px.line(df_scores, x='timestamp', y='score', 
                                    title="Integrity Score Over Time (Simulated)")
                fig_scores.update_layout(height=300)
                st.plotly_chart(fig_scores, use_container_width=True)
        
        with col_anal2:
            # Server Statistics
            st.markdown("#### Server Statistics")
            if stats_data:
                stats = stats_data.get('statistics', {})
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Total Predictions", stats.get('total_predictions', 0))
                    st.metric("Today's Predictions", stats.get('today_predictions', 0))
                
                with col_stat2:
                    st.metric("Active Alerts", stats.get('active_alerts', 0))
                    st.metric("Models Loaded", "2/2" if stats_data.get('models', {}).get('speech_recognition') and stats_data.get('models', {}).get('computer_vision') else "1/2")
                
                # Prediction distribution
                st.markdown("#### Prediction Distribution")
                if 'predictions_by_label' in stats:
                    labels = list(stats['predictions_by_label'].keys())
                    counts = list(stats['predictions_by_label'].values())
                    
                    df_dist = pd.DataFrame({'Label': labels, 'Count': counts})
                    fig_dist = px.pie(df_dist, values='Count', names='Label', 
                                     title="Prediction Distribution")
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Export Data
            st.markdown("#### Export Data")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📥 Export as CSV"):
                    # Prepare data for export
                    export_data = {
                        'latest_predictions': latest_data,
                        'history': history_data.get('history', [])[:50],
                        'alerts': alerts_data.get('alerts', []),
                        'statistics': stats_data,
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    # Convert to DataFrame
                    df_export = pd.DataFrame({
                        'Type': ['Latest', 'History', 'Alerts', 'Stats'],
                        'Count': [
                            1,
                            len(export_data['history']),
                            len(export_data['alerts']),
                            len(export_data['statistics'])
                        ],
                        'Data': [json.dumps(export_data[key]) for key in ['latest_predictions', 'history', 'alerts', 'statistics']]
                    })
                    
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"argus_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col_exp2:
                if st.button("📊 Generate Report"):
                    # Create a summary report
                    report = f"""
                    Argus AI Exam Monitoring - Analytics Report
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    SUMMARY STATISTICS:
                    - Integrity Score: {integrity_score:.1f} ({risk_level})
                    - Active Alerts: {alerts_data.get('count', 0)}
                    - Total Predictions: {stats.get('total_predictions', 0) if stats else 0}
                    - Today's Predictions: {stats.get('today_predictions', 0) if stats else 0}
                    
                    CURRENT PREDICTIONS:
                    - Speech: {speech_pred.replace('_', ' ').title()} ({speech_conf*100:.1f}%)
                    - Vision: {vision_pred.replace('_', ' ').title()} ({vision_conf*100:.1f}%)
                    
                    RECOMMENDATIONS:
                    """
                    
                    if integrity_score < 70:
                        report += "- ⚠️  Integrity score below threshold. Monitor closely.\n"
                    if active_alerts > 0:
                        report += f"- ⚠️  {active_alerts} active alerts require attention.\n"
                    if speech_pred == 'whispering':
                        report += "- 🎤 Whispering detected. Investigate audio activity.\n"
                    if 'suspicious' in vision_pred or 'cheating' in vision_pred:
                        report += "- 👁️ Suspicious visual activity detected.\n"
                    
                    st.text_area("Analytics Report", report, height=300)
    
    with tab5:
        # Alerts Tab
        st.markdown("### ⚠️ Alert Management")
        
        # Alerts summary
        col_alert1, col_alert2, col_alert3 = st.columns(3)
        with col_alert1:
            total_alerts = alerts_data.get('count', 0)
            st.metric("Total Alerts", total_alerts)
        with col_alert2:
            unresolved = sum(1 for a in alerts_data.get('alerts', []) if not a.get('resolved', False))
            st.metric("Unresolved", unresolved)
        with col_alert3:
            high_severity = sum(1 for a in alerts_data.get('alerts', []) if a.get('severity') == 'high')
            st.metric("High Severity", high_severity)
        
        # Alerts list
        st.markdown("#### Recent Alerts")
        if alerts_data and 'alerts' in alerts_data and alerts_data['alerts']:
            for alert in alerts_data['alerts']:
                severity = alert.get('severity', 'medium')
                severity_class = {
                    'high': 'alert-high',
                    'medium': 'alert-medium',
                    'low': 'alert-low'
                }.get(severity, 'alert-medium')
                
                st.markdown(f"""
                <div class="alert-card {severity_class}">
                    <strong>{alert.get('alert_type', 'Unknown').replace('_', ' ').title()}</strong><br>
                    <small>Severity: {severity.upper()} | Device: {alert.get('device_id', 'Unknown')}</small><br>
                    {alert.get('description', 'No description')}<br>
                    <small>{alert.get('timestamp', 'Unknown time')}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Resolve button
                col_alert_btn1, col_alert_btn2 = st.columns([4, 1])
                with col_alert_btn2:
                    if st.button("✅ Resolve", key=f"resolve_{alert.get('id')}"):
                        st.success(f"Alert {alert.get('id')} marked as resolved")
                        # In a real implementation, you would call an API to update the alert
        else:
            st.success("🎉 No active alerts! All systems are normal.")
        
        # Alert filters
        st.markdown("#### Alert Filters")
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            show_resolved = st.checkbox("Show Resolved Alerts")
        with col_filter2:
            severity_filter = st.multiselect(
                "Filter by Severity",
                ['high', 'medium', 'low'],
                default=['high', 'medium']
            )
        
        # Create new alert (for testing)
        st.markdown("#### Create Test Alert")
        with st.expander("Create New Alert"):
            alert_type = st.selectbox("Alert Type", [
                "whispering_detected", 
                "suspicious_behavior", 
                "device_offline",
                "high_integrity_risk"
            ])
            severity = st.selectbox("Severity", ["low", "medium", "high"])
            description = st.text_area("Description")
            
            if st.button("Create Test Alert"):
                st.warning(f"Test alert created: {alert_type}")
                # In a real implementation, you would call an API to create an alert
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns([1, 2, 1])
    with col_footer2:
        st.markdown(f"""
        <div style="text-align: center; color: #6B7280; padding: 1rem;">
            <p><strong>👁️ Argus - Integrity Through Intelligent Vision</strong></p>
            <p>Samsung Innovation Campus Batch 7 Stage 3 | Team Sicat | BINUS University</p>
            <p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Student: {students.get(st.session_state.selected_student, 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Show refresh timer in sidebar
        time_remaining = max(0, refresh_interval - time_since_refresh)
        st.sidebar.markdown(f"**Next refresh in:** {int(time_remaining)}s")
    
    # Cleanup on exit
    if not st.session_state.is_camera_on and st.session_state.camera.cap:
        st.session_state.camera.stop_capture()

# =====================================================================
# RUN DASHBOARD
# =====================================================================
if __name__ == "__main__":
    # Add autorefresh if needed
    # count = st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")
    
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check your FastAPI server connection and try again.")