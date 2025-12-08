# test_api.py
import requests
import time

# Test the latest prediction endpoint
def test_latest_endpoint():
    response = requests.get("http://localhost:5000/latest")
    print("Latest prediction:", response.json())
    return response.json()

# Test file upload endpoint
def test_upload_endpoint(audio_file_path):
    with open(audio_file_path, 'rb') as f:
        files = {'file': (audio_file_path.split('/')[-1], f, 'audio/wav')}
        response = requests.post("http://localhost:5000/upload", files=files)
    print("Upload response:", response.json())
    return response.json()

if __name__ == "__main__":
    # Test 1: Check latest prediction
    print("Testing /latest endpoint...")
    test_latest_endpoint()
    
    # Test 2: Upload an audio file
    print("\nTesting /upload endpoint...")
    # Replace with your actual audio file path
    test_upload_endpoint("path/to/your/audio.wav")
    
    # Test 3: Check if latest prediction was updated
    time.sleep(1)
    print("\nChecking updated prediction...")
    test_latest_endpoint()