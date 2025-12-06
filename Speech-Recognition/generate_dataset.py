import librosa
import numpy as np
import pandas as pd
import os
import glob

# ================= KONFIGURASI =================
DATASET_PATH = "dataset_audio"  # Nama folder utama
OUTPUT_CSV = "argus_audio_dataset.csv"
SAMPLE_RATE = 22050  # Standar librosa
CHUNK_DURATION = 1.0 # Durasi per potong (detik). 1 Baris CSV = 1 Detik audio.
# ===============================================

def extract_features_from_chunk(y, sr, timestamp_start):
    """
    Mengekstrak fitur spesifik dari potongan audio (chunk)
    """
    # 1. RMS Amplitude (Kekerasan Suara)
    rms = np.mean(librosa.feature.rms(y=y))
    
    # 2. Peak Amplitude (Amplitudo Tertinggi)
    peak_amplitude = np.max(np.abs(y))
    
    # 3. Zero Crossing Rate (Tingkat perubahan tanda sinyal - bagus untuk deteksi desis/bisik)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # 4. Spectral Centroid (Titik berat spektrum - 'kecerahan' suara)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # 5. MFCC (Mel-frequency cepstral coefficients) - Fitur standar speech recognition
    # Kita ambil 13 koefisien, tapi nanti cuma simpan 3 teratas sesuai request
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1) # Rata-rata tiap koefisien
    
    # Menyusun data sesuai request user
    features = {
        "timestamp": timestamp_start,
        "rms_amplitude": rms,
        "peak_amplitude": peak_amplitude,
        "zero_cross_rate": zcr,
        "spectral_centroid": spectral_centroid,
        "mfcc_1": mfcc_mean[0], # Koefisien ke-1
        "mfcc_2": mfcc_mean[1], # Koefisien ke-2
        "mfcc_3": mfcc_mean[2]  # Koefisien ke-3
    }
    
    return features

def process_dataset():
    all_data = []
    
    # Cek apakah folder ada
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Folder '{DATASET_PATH}' tidak ditemukan. Buat folder dulu!")
        return

    # List semua label (nama sub-folder)
    labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    print(f"Label ditemukan: {labels}")
    
    for label in labels:
        folder_path = os.path.join(DATASET_PATH, label)
        # Ambil semua file audio (bisa m4a, wav, mp3)
        audio_files = glob.glob(os.path.join(folder_path, "*.*"))
        
        print(f"Processing label: '{label}' ({len(audio_files)} files)...")
        
        for file_path in audio_files:
            try:
                # Load audio file
                # librosa otomatis convert m4a jika ffmpeg terinstall
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Hitung jumlah sampel per chunk
                samples_per_chunk = int(CHUNK_DURATION * sr)
                total_samples = len(y)
                
                # Sliding Window: Potong audio jadi bagian-bagian kecil
                for i in range(0, total_samples, samples_per_chunk):
                    chunk = y[i:i+samples_per_chunk]
                    
                    # Pastikan chunk tidak terlalu pendek (misal sisa akhir file)
                    if len(chunk) < samples_per_chunk:
                        continue
                        
                    # Hitung timestamp (detik ke-berapa chunk ini dimulai)
                    timestamp_start = i / sr
                    
                    # Ekstrak fitur
                    features = extract_features_from_chunk(chunk, sr, timestamp_start)
                    features["label"] = label # Tambahkan label
                    
                    all_data.append(features)
                    
            except Exception as e:
                print(f"Gagal memproses {file_path}: {e}")

    # Convert ke Pandas DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder kolom agar 'label' ada di paling kanan (Best Practice)
    cols = ["timestamp", "rms_amplitude", "peak_amplitude", "zero_cross_rate", 
            "spectral_centroid", "mfcc_1", "mfcc_2", "mfcc_3", "label"]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSUKSES! Dataset tersimpan di: {OUTPUT_CSV}")
    print(f"Total Data: {len(df)} baris")
    print(df.head())

if __name__ == "__main__":
    process_dataset()