import streamlit as st
import requests
import time

SERVER = "http://192.168.1.10:5000/latest"

st.title("ARGUS Speech Recognition Monitor")
st.subheader("Real-Time Audio Classification")

placeholder = st.empty()

while True:
    try:
        r = requests.get(SERVER).json()
        label = r["label"]

        placeholder.metric("Current Audio Class", label.upper())

    except:
        st.error("Failed to connect to server.")

    time.sleep(0.5)