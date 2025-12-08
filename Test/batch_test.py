# batch_test.py
import requests
import os
import json
from tqdm import tqdm

def batch_test_audio_files(folder_path):
    """Test multiple audio files from a folder"""
    results = []
    
    # Get all WAV files in folder
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    print(f"Found {len(audio_files)} audio files for testing")
    
    for audio_file in tqdm(audio_files):
        file_path = os.path.join(folder_path, audio_file)
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (audio_file, f, 'audio/wav')}
                response = requests.post(
                    "http://localhost:5000/upload", 
                    files=files,
                    timeout=30
                )
            
            result = {
                'file': audio_file,
                'response': response.json(),
                'status': 'success' if response.status_code == 200 else 'failed'
            }
            results.append(result)
            
        except Exception as e:
            results.append({
                'file': audio_file,
                'error': str(e),
                'status': 'error'
            })
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nTest Summary:")
    print(f"Total files: {len(audio_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(audio_files) - success_count}")
    
    return results

if __name__ == "__main__":
    # Test with your audio samples folder
    batch_test_audio_files("./test_audio_samples/")