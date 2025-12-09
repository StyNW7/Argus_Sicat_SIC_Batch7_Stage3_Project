<div align="center">
    <div>
        <img height="150px" src="./Images/Logo.png" alt="Argus Logo"/>
    </div>
    <div>
            <h3><b>Argus</b></h3>
            <p><i>Integrity Through Intelligent Vision 👁️</i></p>
    </div>
</div>
<br>
<h1 align="center">Argus - Sicat Team - Samsung Innovation Campus Batch 7 - Stage 3</h1>
<div align="center">

<img src="./Images/Preview/Preview.jpg" alt="Argus Device Preview"/>

</div>
<br>

Argus is a next-generation AI + IoT proctoring system designed to maintain academic integrity through real-time intelligent monitoring.

It integrates Computer Vision, Speech Recognition, and IoT sensors to detect abnormal user behavior such as whispering, unauthorized talking, suspicious movements, and so on.

Video Demo Link: 
...

---

## ⚙️ Technology Stack

<div align="center">

<kbd><img src="./Images/esp32.png" height="60" /></kbd>
<kbd><img src="./Images/cpp.png" height="60" /></kbd>
<kbd><img src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/arduino.png" height="60" /></kbd>
<kbd><img src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/python.png" height="60" /></kbd>
<kbd><img src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/streamlit.png" height="60" /></kbd>

</div>

<div align="center">
<h4>ESP32 | C++ (.ino) | Python | Computer Vision | Speech Recognition | Streamlit</h4>
</div>

---

## 🧩 Core Features

### 👁️ Real-Time Computer Vision Monitoring

Argus uses a trained ResNet-based CNN model to detect:
- Suspicious head/eye movements
- Unauthorized presence
- Strange posture or look-away events
- Possible cheating behaviors

🔍 *AI-powered visual integrity.*

---

### 🎙️ Whispering & Speech Detection (Audio AI)

Argus integrates sound-based classification with 3 custom labels:
- Silence (Clean No Sound)
- Whispering (Suspicious)
- Normal Conversation

This allows the system to differentiate harmless dialogue from cheating behavior.

🎧 *Noise-aware, environment-aware speech intelligence.*

---

### 🌐 IoT Camera & Microphone Data Streaming

The ESP32 sends camera frames and audio packets to the backend using:
- HTTP streaming
- WiFi-based segmented audio uploads

Backend runs AI inference on the received data and updates the dashboard in real-time.

📡 *Lightweight, responsive IoT integration.*

---

### 📊 Streamlit Real-Time Dashboard

Argus features a modern, clean dashboard including:
- Live camera feed
- Real-time speech recognition output
- Action logs & event alerts
- System status monitoring

📈 *Fast & interactive monitoring experience.*

---

### 🌩 Backend with FastAPI

FastAPI handles:
- Image inference endpoint (/vision)
- Audio classification endpoint (/upload)
- Latest prediction retrieval (/latest)
- Real-time streaming compatibility

⚡ *Secure and ultra-fast backend processing.*

---

## 🧰 Getting Started

### Hardware Requirements

- ESP32 Microcontroller
- ESP32-Cam
- INMP411 Ominichannel Microphone
- LED
- Buzzer
- Push Button
- Breadboard + Jumper Cables

---

## 🚀 Software Workflow

1. ESP32 captures camera frames → Sends to FastAPI endpoint
2. ESP32 microphone records audio segments → Uploads WAV packets
3. FastAPI performs:
    - Computer Vision inference (ResNet)
    - Speech Recognition inference (MLP/RandomForest)
4. Streamlit dashboard pulls the latest AI results
5. System displays:
    - Alerts
    - Labels
    - Logs
    - Live monitoring

---

## 📸 &nbsp;Result Preview
<table style="width:100%; text-align:center">
    <col width="100%">
    <tr>
        <td width="1%" align="center"><img height="370" src="./Images/Preview/1.jpg"/></td>
    </tr>
    <tr>
        <td width="1%" align="center">Output at Arduino IDE</td>
    </tr>
    <tr>
        <td width="1%" align="center"><img height="400" src="./Images/Preview/2.jpg"/></td>
    </tr>
    <tr>
        <td width="1%" align="center">Samsung IoT Tool Kit</td>
    </tr>
    <tr>
        <td width="1%" align="center"><img height="400" src="./Images/Preview/3.JPG"/></td>
    </tr>
    <tr>
        <td width="1%" align="center">OLED Display Output</td>
    </tr>
    <tr>
        <td width="1%" align="center"><img height="400" src="./Images/Preview/4.JPG"/></td>
    </tr>
    <tr>
        <td width="1%" align="center">Argus IoT Preview</td>
    </tr>
    <tr>
        <td width="1%" align="center"><img height="400" src="./Images/Preview/5.jpg"/></td>
    </tr>
    <tr>
        <td width="1%" align="center">Completing the Project</td>
    </tr>
</table>

---

## 🧭 Diagram

*Overall Block Components Diagram*
<p align="center">
  <img src="./Images/Diagram/block-diagram.png" width="700">
</p>

This diagram illustrates how the IoT devices, AI inference modules, backend server, and dashboard interact within the Argus ecosystem.

---

## 👥 Owner

This Repository is created by Team Sicat - Samsung Innovation Campus
<ul>
<li>Stanley Nathanael Wijaya - Team Leader</li>
<li>Clarissa Aditjakra</li>
<li>Jazzlyn Amelia Lim</li>
<li>Visella</li>
</ul>
As Final Project for SIC Batch 7 Stage 3

---

## 📬 Contact
Have questions or want to collaborate?

- 📧 Email: stanley.n.wijaya7@gmail.com
- 💬 Discord: `stynw7`

<code>Made with ❤️ by Team Sicat</code>