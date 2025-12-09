# streamlit_dashboard_esp32.py
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
import io
import base64
import warnings
from datetime import datetime, timedelta
import asyncio
import websockets
import sqlite3
from collections import deque
import altair as alt
import schedule
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Argus - IoT Exam Monitoring",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .device-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .alert-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 5px 15px;
        border-radius: 20px;
        color: white;
        display: inline-block;
        font-weight: bold;
    }
    .status-dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .status-online {
        background-color: #10B981;
    }
    .status-offline {
        background-color: #EF4444;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
FASTAPI_URL = "http://localhost:5000"  # Change to your server IP
DATABASE_PATH = "argus_data.db"

class IoTDeviceManager:
    def __init__(self, api_url):
        self.api_url = api_url
        self.devices = {}
        self.last_update = None
        
    def fetch_devices(self):
        """Fetch connected devices from API"""
        try:
            response = requests.get(f"{self.api_url}/devices", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.devices = data.get('devices', {})
                self.last_update = datetime.now()
                return True
        except Exception as e:
            st.error(f"Error fetching devices: {e}")
        return False
    
    def get_device_history(self, device_id, limit=50):
        """Get prediction history for a device"""
        try:
            response = requests.get(
                f"{self.api_url}/device/{device_id}/history?limit={limit}",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching device history: {e}")
        return None
    
    def get_latest_predictions(self):
        """Get latest predictions for all devices"""
        try:
            response = requests.get(f"{self.api_url}/latest", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")
        return {}
    
    def get_alerts(self, limit=20):
        """Get recent alerts"""
        try:
            response = requests.get(f"{self.api_url}/alerts?limit={limit}", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching alerts: {e}")
        return {}
    
    def get_stats(self):
        """Get server statistics"""
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching stats: {e}")
        return {}

def init_database():
    """Initialize local database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create local cache table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dashboard_cache (
        key TEXT PRIMARY KEY,
        value TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def calculate_integrity_score(predictions_data):
    """Calculate integrity score based on recent predictions"""
    if not predictions_data:
        return 100
    
    # Count whispering events in last 10 predictions
    whispering_count = 0
    total_predictions = 0
    
    for device_id, prediction in predictions_data.get('devices', {}).items():
        if isinstance(prediction, dict) and prediction.get('prediction') == 'whispering':
            whispering_count += 1
        total_predictions += 1
    
    if total_predictions == 0:
        return 100
    
    # Calculate score: 100 - (whispering_percentage * 40)
    whispering_percentage = (whispering_count / total_predictions) * 100
    integrity_score = 100 - (whispering_percentage * 0.4)
    
    return max(0, min(100, integrity_score))

def get_risk_level(score):
    """Determine risk level based on integrity score"""
    if score >= 70:
        return "Safe", "🟢", "success"
    elif score >= 35:
        return "Alert", "🟡", "warning"
    else:
        return "Warning", "🔴", "error"

def create_device_status_chart(devices):
    """Create chart showing device status"""
    online_count = sum(1 for device in devices.values() 
                      if device.get('status') == 'active')
    offline_count = len(devices) - online_count
    
    fig = go.Figure(data=[go.Pie(
        labels=['Online', 'Offline'],
        values=[online_count, offline_count],
        hole=.3,
        marker_colors=['#10B981', '#EF4444']
    )])
    
    fig.update_layout(
        title="Device Status",
        height=300,
        showlegend=True
    )
    
    return fig

def create_prediction_timeline(device_history):
    """Create timeline chart for device predictions"""
    if not device_history or 'history' not in device_history:
        return go.Figure()
    
    history = device_history['history']
    df = pd.DataFrame(history)
    
    if 'timestamp' not in df.columns or 'prediction' not in df.columns:
        return go.Figure()
    
    df['time'] = pd.to_datetime(df['timestamp'])
    
    color_map = {
        'silence': '#3B82F6',
        'whispering': '#EF4444',
        'normal_conversation': '#10B981'
    }
    
    fig = px.scatter(df, x='time', y='prediction',
                     color='prediction',
                     color_discrete_map=color_map,
                     title="Prediction Timeline",
                     labels={'prediction': 'Audio Type', 'time': 'Time'})
    
    fig.update_layout(height=300, showlegend=True)
    return fig

def create_confidence_gauge(confidence):
    """Create gauge chart for prediction confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))
    fig.update_layout(height=250)
    return fig

def main():
    # Initialize session state
    if 'device_manager' not in st.session_state:
        st.session_state.device_manager = IoTDeviceManager(FASTAPI_URL)
    
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = None
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Initialize database
    init_database()
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Argus - IoT Exam Monitoring Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h4 class="sub-header">Real-time monitoring of exam integrity using ESP32 devices with AI-powered speech recognition</h4>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Server Configuration
        st.markdown("### 🔧 Server Configuration")
        server_url = st.text_input("FastAPI Server URL", FASTAPI_URL)
        if server_url != FASTAPI_URL:
            st.session_state.device_manager.api_url = server_url
        
        # Auto-refresh
        st.markdown("### 🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
        st.session_state.auto_refresh = auto_refresh
        
        if st.button("🔄 Manual Refresh"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Device Management
        st.markdown("### 📱 Device Management")
        
        # Register new device
        with st.expander("➕ Register New Device"):
            new_device_id = st.text_input("Device ID")
            new_student_id = st.text_input("Student ID")
            new_ip = st.text_input("IP Address (optional)")
            
            if st.button("Register Device"):
                try:
                    response = requests.post(
                        f"{server_url}/device/{new_device_id}/register",
                        params={"student_id": new_student_id, "ip_address": new_ip}
                    )
                    if response.status_code == 200:
                        st.success("Device registered successfully!")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error registering device: {e}")
        
        # Connection Status
        st.markdown("### 🔗 Connection Status")
        try:
            response = requests.get(f"{server_url}/stats", timeout=2)
            if response.status_code == 200:
                st.success("✅ Server Connected")
                stats = response.json()
                st.info(f"Uptime: {stats.get('uptime', 0):.0f}s")
            else:
                st.error("❌ Server Error")
        except:
            st.error("❌ Cannot connect to server")
    
    # Main Dashboard - Top Metrics
    st.markdown("---")
    
    # Fetch data
    device_manager = st.session_state.device_manager
    device_manager.fetch_devices()
    
    latest_predictions = device_manager.get_latest_predictions()
    alerts = device_manager.get_alerts()
    stats = device_manager.get_stats()
    
    # Calculate metrics
    integrity_score = calculate_integrity_score(latest_predictions)
    risk_level, risk_emoji, risk_color = get_risk_level(integrity_score)
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="device-card">
            <p class="metric-label">Connected Devices</p>
            <p class="metric-value">{}</p>
        </div>
        """.format(len(device_manager.devices)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="device-card">
            <p class="metric-label">Integrity Score</p>
            <p class="metric-value">{:.1f}</p>
            <p>{} {}</p>
        </div>
        """.format(integrity_score, risk_emoji, risk_level), unsafe_allow_html=True)
    
    with col3:
        total_alerts = alerts.get('count', 0)
        unresolved = alerts.get('unresolved_count', 0)
        st.markdown("""
        <div class="device-card">
            <p class="metric-label">Active Alerts</p>
            <p class="metric-value">{}</p>
            <p>({} unresolved)</p>
        </div>
        """.format(total_alerts, unresolved), unsafe_allow_html=True)
    
    with col4:
        total_preds = stats.get('total_predictions', 0)
        st.markdown("""
        <div class="device-card">
            <p class="metric-label">Total Predictions</p>
            <p class="metric-value">{}</p>
            <p>Today: {}</p>
        </div>
        """.format(total_preds, stats.get('today_predictions', 0)), unsafe_allow_html=True)
    
    # Device Monitoring Section
    st.markdown("---")
    st.markdown("## 📱 Connected Devices")
    
    if device_manager.devices:
        # Create device selection
        device_ids = list(device_manager.devices.keys())
        
        if not st.session_state.selected_device and device_ids:
            st.session_state.selected_device = device_ids[0]
        
        selected_device = st.selectbox(
            "Select Device to Monitor",
            device_ids,
            index=0 if not st.session_state.selected_device else device_ids.index(st.session_state.selected_device)
        )
        
        if selected_device:
            st.session_state.selected_device = selected_device
            
            # Get device details
            device_info = device_manager.devices[selected_device]
            device_history = device_manager.get_device_history(selected_device)
            
            # Device Details Card
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6;">
                    <h3>Device: {selected_device}</h3>
                    <p><strong>Student ID:</strong> {device_info.get('student_id', 'N/A')}</p>
                    <p><strong>IP Address:</strong> {device_info.get('ip_address', 'N/A')}</p>
                    <p><strong>Last Seen:</strong> {device_info.get('last_seen', 'N/A')}</p>
                    <p><strong>Status:</strong> 
                        <span class="status-dot status-online"></span>
                        {device_info.get('status', 'unknown')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Latest prediction for this device
                device_prediction = latest_predictions.get('devices', {}).get(selected_device, {})
                if device_prediction:
                    pred_label = device_prediction.get('prediction', 'unknown').replace('_', ' ').title()
                    confidence = device_prediction.get('confidence', 0)
                    
                    color_map = {
                        'silence': '#3B82F6',
                        'whispering': '#EF4444',
                        'normal_conversation': '#10B981'
                    }
                    color = color_map.get(device_prediction.get('prediction'), '#6B7280')
                    
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: {color}; color: white;">
                        <h4>Current Prediction</h4>
                        <h2>{pred_label}</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                # Prediction probabilities
                if device_prediction and 'probabilities' in device_prediction:
                    probs = device_prediction['probabilities']
                    if isinstance(probs, dict):
                        df_probs = pd.DataFrame({
                            'Label': list(probs.keys()),
                            'Probability': list(probs.values())
                        })
                        df_probs['Label'] = df_probs['Label'].str.replace('_', ' ').str.title()
                        
                        fig_probs = px.bar(df_probs, x='Label', y='Probability',
                                          color='Label',
                                          color_discrete_map={
                                              'Silence': '#3B82F6',
                                              'Whispering': '#EF4444',
                                              'Normal Conversation': '#10B981'
                                          })
                        fig_probs.update_layout(height=200, showlegend=False)
                        st.plotly_chart(fig_probs, use_container_width=True)
            
            # Charts for selected device
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Confidence Gauge
                if device_prediction:
                    confidence = device_prediction.get('confidence', 0)
                    fig_gauge = create_confidence_gauge(confidence)
                    st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_chart2:
                # Prediction Timeline
                fig_timeline = create_prediction_timeline(device_history)
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Feature Visualization
            st.markdown("### 🎵 Audio Feature Analysis")
            
            if device_prediction and 'features' in device_prediction:
                features = device_prediction['features']
                if isinstance(features, list) and len(features) > 0:
                    # Create feature names (based on your feature extraction)
                    feature_names = ['RMS', 'ZCR', 'Spectral Centroid'] + [f'MFCC {i+1}' for i in range(13)]
                    
                    df_features = pd.DataFrame({
                        'Feature': feature_names[:len(features)],
                        'Value': features[:len(feature_names)]
                    })
                    
                    # Feature importance chart
                    fig_features = px.bar(df_features, x='Feature', y='Value',
                                         title="Audio Feature Values")
                    fig_features.update_layout(height=300)
                    st.plotly_chart(fig_features, use_container_width=True)
    else:
        st.info("No devices connected. Connect an ESP32 device to begin monitoring.")
    
    # All Devices Overview
    st.markdown("---")
    st.markdown("## 📊 All Devices Overview")
    
    if device_manager.devices:
        # Devices table
        devices_list = []
        for device_id, info in device_manager.devices.items():
            # Get latest prediction for each device
            pred = latest_predictions.get('devices', {}).get(device_id, {})
            
            devices_list.append({
                'Device ID': device_id,
                'Student ID': info.get('student_id', 'N/A'),
                'Status': info.get('status', 'unknown'),
                'Last Prediction': pred.get('prediction', 'unknown').replace('_', ' ').title(),
                'Confidence': f"{pred.get('confidence', 0):.1%}" if pred.get('confidence') else 'N/A',
                'Last Seen': info.get('last_seen', 'N/A')
            })
        
        df_devices = pd.DataFrame(devices_list)
        st.dataframe(df_devices, use_container_width=True)
        
        # Charts row
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Device status chart
            fig_status = create_device_status_chart(device_manager.devices)
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col_chart2:
            # Prediction distribution
            if latest_predictions.get('devices'):
                pred_counts = {}
                for device_id, pred in latest_predictions['devices'].items():
                    if isinstance(pred, dict):
                        label = pred.get('prediction', 'unknown')
                        pred_counts[label] = pred_counts.get(label, 0) + 1
                
                if pred_counts:
                    df_pred_dist = pd.DataFrame({
                        'Label': list(pred_counts.keys()),
                        'Count': list(pred_counts.values())
                    })
                    df_pred_dist['Label'] = df_pred_dist['Label'].str.replace('_', ' ').str.title()
                    
                    fig_dist = px.pie(df_pred_dist, values='Count', names='Label',
                                     title="Prediction Distribution",
                                     color_discrete_sequence=px.colors.sequential.RdBu)
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Waiting for devices to connect...")
    
    # Alerts Section
    st.markdown("---")
    st.markdown("## ⚠️ Recent Alerts")
    
    if alerts.get('alerts'):
        alerts_df = pd.DataFrame(alerts['alerts'])
        
        # Display alerts in expandable sections by severity
        for severity in ['high', 'medium', 'low']:
            severity_alerts = alerts_df[alerts_df['severity'] == severity]
            if not severity_alerts.empty:
                with st.expander(f"{severity.title()} Severity Alerts ({len(severity_alerts)})", expanded=severity=='high'):
                    for _, alert in severity_alerts.iterrows():
                        alert_color = {
                            'high': 'red',
                            'medium': 'orange',
                            'low': 'yellow'
                        }.get(severity, 'gray')
                        
                        st.markdown(f"""
                        <div style="border-left: 5px solid {alert_color}; 
                                    padding: 10px; 
                                    margin: 5px 0; 
                                    background-color: #f8f9fa;">
                            <strong>{alert['alert_type'].replace('_', ' ').title()}</strong><br>
                            Device: {alert['device_id']}<br>
                            {alert['description']}<br>
                            <small>{alert['timestamp']}</small>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.success("No active alerts. All systems normal.")
    
    # Export and Reports
    st.markdown("---")
    st.markdown("## 📊 Reports & Export")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📥 Export All Data"):
            # Combine all data
            all_data = {
                'devices': device_manager.devices,
                'latest_predictions': latest_predictions,
                'alerts': alerts,
                'stats': stats,
                'export_time': datetime.now().isoformat()
            }
            
            # Convert to JSON
            json_data = json.dumps(all_data, indent=2, default=str)
            
            st.download_button(
                label="Download JSON Report",
                data=json_data,
                file_name=f"argus_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col_export2:
        if st.button("📈 Generate Summary Report"):
            # Create summary report
            summary = f"""
            Argus Exam Monitoring - Summary Report
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Overall Integrity Score: {integrity_score:.1f} ({risk_level})
            Connected Devices: {len(device_manager.devices)}
            Total Predictions: {stats.get('total_predictions', 0)}
            Active Alerts: {alerts.get('count', 0)}
            
            Device Status:
            """
            
            for device_id, info in device_manager.devices.items():
                pred = latest_predictions.get('devices', {}).get(device_id, {})
                summary += f"\n- {device_id}: {info.get('student_id')} | {pred.get('prediction', 'unknown')}"
            
            st.text_area("Summary Report", summary, height=200)
    
    with col_export3:
        if st.button("🔄 Reset Dashboard"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'device_manager':
                    del st.session_state[key]
            st.rerun()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Show refresh countdown
        time_remaining = refresh_interval - time_since_refresh
        st.sidebar.markdown(f"**Next refresh in:** {int(time_remaining)}s")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>👁️ <strong>Argus - Integrity Through Intelligent Vision</strong></p>
        <p>IoT + AI Exam Monitoring System | Team Sicat | Samsung Innovation Campus</p>
        <p>Last Updated: {} | Server Status: {}</p>
    </div>
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "✅ Online" if stats else "❌ Offline"
    ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()