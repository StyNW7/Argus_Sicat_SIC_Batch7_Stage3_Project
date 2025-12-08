import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import cv2
import plotly.graph_objects as go
import threading
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import queue

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Argus Proctoring System",
    page_icon="👁️",
    layout="wide",
)

# API Configuration
API_URL = "http://127.0.0.1:5000/upload"
SAMPLE_RATE = 16000  # Matches your librosa load
DURATION = 2.0       # Seconds per audio chunk (window size)

# Global Variables for Threading
# We use a Queue to pass data from the Audio Thread to the Main UI Thread safely
audio_result_queue = queue.Queue()
stop_threads = False

# ==========================================
# 2. CUSTOM CSS (Clean Dark Mode)
# ==========================================
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #414449;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS & THREADING
# ==========================================

def calculate_integrity(current_score, vision_label, audio_label):
    """Updates the integrity score based on violations."""
    penalty = 0
    
    # Audio Penalties
    if audio_label == "whispering":
        penalty += 2.0
    elif audio_label == "normal_conversation":
        penalty += 1.0  # Talking during exam is also suspicious
        
    # Vision Penalties (Placeholder for future update)
    if vision_label == "Suspicious":
        penalty += 0.5
        
    new_score = max(0, min(100, current_score - penalty))
    
    # Healing (Regenerate score if behaving well)
    if audio_label == "silence" and vision_label == "Safe":
        new_score = min(100, new_score + 0.1)
        
    return new_score

def audio_recorder_thread():
    """
    Background thread that records audio continuously 
    and sends it to the FastAPI server.
    """
    global stop_threads
    print("🎙️ Audio Thread Started...")
    
    while not stop_threads:
        try:
            # 1. Record Audio (Blocking for DURATION seconds)
            # This happens in background, so Video won't freeze!
            recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                             samplerate=SAMPLE_RATE, channels=1)
            sd.wait() # Wait until recording is finished
            
            # 2. Save to temp file
            temp_filename = "temp_dashboard_rec.wav"
            write(temp_filename, SAMPLE_RATE, recording)
            
            # 3. Send to FastAPI
            with open(temp_filename, 'rb') as f:
                files = {'file': (temp_filename, f, 'audio/wav')}
                response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                # Put result in queue for the Main Thread to read
                audio_result_queue.put(result)
            else:
                print("API Error:", response.status_code)
                
        except Exception as e:
            print(f"Audio Thread Error: {e}")
            time.sleep(1) # Prevent tight loop on error

# ==========================================
# 4. SESSION STATE INIT
# ==========================================
if 'integrity_score' not in st.session_state:
    st.session_state['integrity_score'] = 100
if 'exam_active' not in st.session_state:
    st.session_state['exam_active'] = False
if 'last_audio_label' not in st.session_state:
    st.session_state['last_audio_label'] = "silence"
if 'history_data' not in st.session_state:
    st.session_state['history_data'] = pd.DataFrame(columns=['timestamp', 'score', 'label'])

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🛡️ Argus Controller")
    st.markdown("---")
    
    if not st.session_state['exam_active']:
        if st.button("▶ START EXAM", type="primary"):
            st.session_state['exam_active'] = True
            stop_threads = False
            # Start the Audio Thread
            t = threading.Thread(target=audio_recorder_thread, daemon=True)
            t.start()
            st.rerun()
    else:
        if st.button("⏹ STOP EXAM", type="secondary"):
            st.session_state['exam_active'] = False
            stop_threads = True
            st.rerun()

    st.markdown("---")
    st.info(f"Backend Status: Connecting to {API_URL}")

# ==========================================
# 6. MAIN DASHBOARD
# ==========================================

# Title & Score
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("👁️ Argus Intelligent Proctoring")
    st.markdown("Multimodal Monitoring: **Vision (Camera)** + **Speech (FastAPI AI)**")
with col_head2:
    score = st.session_state['integrity_score']
    st.metric("Integrity Score", f"{score:.1f}%", delta=f"{score-100:.1f}")

# Layout
col_video, col_stats = st.columns([2, 1])

# A. VIDEO FEED (Computer Vision Placeholder)
with col_video:
    st.subheader("📷 Live Vision Feed")
    camera_placeholder = st.empty()

# B. AUDIO & STATS AREA
with col_stats:
    st.subheader("🎙️ Acoustic Analysis")
    audio_status_ph = st.empty()
    st.markdown("---")
    st.subheader("📊 Timeline")
    chart_ph = st.empty()

# ==========================================
# 7. MAIN LOOP
# ==========================================
if st.session_state['exam_active']:
    # Open Camera
    cap = cv2.VideoCapture(0)
    
    while st.session_state['exam_active']:
        # --- 1. PROCESS VISION (Main Thread) ---
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Placeholder Logic for Vision (Update this later with your CV model)
                # For now, let's assume Vision is Safe unless updated
                vision_label = "Safe" 
                
                # Draw Overlay
                cv2.putText(frame, f"VISION: {vision_label}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display Frame
                camera_placeholder.image(frame, channels="RGB", use_column_width=True)

        # --- 2. PROCESS AUDIO (Check Queue from Background Thread) ---
        try:
            # Try to get data from the thread (non-blocking)
            # We assume the thread puts data in queue every ~2 seconds
            while not audio_result_queue.empty():
                result = audio_result_queue.get_nowait()
                st.session_state['last_audio_label'] = result.get("prediction", "error")
        except queue.Empty:
            pass
        
        current_audio_label = st.session_state['last_audio_label']

        # --- 3. UPDATE LOGIC & UI ---
        
        # Calculate Score
        st.session_state['integrity_score'] = calculate_integrity(
            st.session_state['integrity_score'], vision_label, current_audio_label
        )
        
        # Update History
        new_row = pd.DataFrame([{
            'timestamp': datetime.now(),
            'score': st.session_state['integrity_score'],
            'label': current_audio_label
        }])
        st.session_state['history_data'] = pd.concat(
            [st.session_state['history_data'], new_row], ignore_index=True
        ).tail(50) # Keep last 50 points
        
        # Visualize Audio Status
        if current_audio_label == "whispering":
            bg_color = "#ff4b4b" # Red
            icon = "⚠️"
        elif current_audio_label == "normal_conversation":
            bg_color = "#ffa421" # Orange
            icon = "🗣️"
        else:
            bg_color = "#21c354" # Green
            icon = "🤫"
            
        audio_status_ph.markdown(f"""
            <div class="status-box" style="background-color: {bg_color}; color: white;">
                <h1>{icon}</h1>
                <h3>{current_audio_label.upper()}</h3>
                <p>Real-time Audio Classification</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualize Chart
        fig = go.Figure()
        hist_df = st.session_state['history_data']
        fig.add_trace(go.Scatter(
            x=hist_df['timestamp'], 
            y=hist_df['score'],
            mode='lines',
            name='Integrity',
            line=dict(color='white', width=3)
        ))
        fig.update_layout(
            height=200, 
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )
        chart_ph.plotly_chart(fig, use_container_width=True)

        # Small sleep to yield control
        time.sleep(0.05)

    cap.release()
else:
    # Standby Screen
    camera_placeholder.image("https://via.placeholder.com/800x450.png?text=Press+START+to+Begin+Exam", use_column_width=True)
    audio_status_ph.info("Audio Module: Standby")