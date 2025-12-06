#!/usr/bin/env python3
"""
ROBUST extract_features.py

Fitur:
- Mendukung semua format audio (m4a, mp3, wav, ogg, flac)
- Jika gagal load → otomatis convert ke wav via FFmpeg
- Logging file sukses/gagal
- Skip file rusak atau terlalu pendek
- Aman digunakan untuk dataset Argus Speech Recognition
"""

import argparse
from pathlib import Path
import subprocess
import librosa
import numpy as np
import pandas as pd
from datetime import datetime
import soundfile as sf


# ========== KONVERSI OTOMATIS M4A → WAV via FFmpeg ==========
def convert_to_wav(input_file, temp_wav):
    try:
        cmd = [
            "ffmpeg",
            "-y",           # overwrite
            "-i", str(input_file),
            "-ac", "1",     # mono
            "-ar", "16000", # 16k sample rate
            str(temp_wav)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"[!] FFmpeg conversion failed: {e}")
        return False


# ========== EKSTRAKSI FITUR PER FRAME ==========
def extract_features_from_signal(y, sr, frame_length_seconds=0.3):
    frame_len = int(sr * frame_length_seconds)
    hop_len = frame_len   # no overlap

    if len(y) < frame_len:
        return []  # skip audio terlalu pendek

    features = []

    for start in range(0, len(y) - frame_len + 1, hop_len):
        frame = y[start:start+frame_len]

        # RMS
        rms = float(np.sqrt(np.mean(frame**2)))

        # ZCR
        try:
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(frame)[0]))
        except Exception:
            zcr = 0.0

        # Spectral Centroid
        try:
            spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=frame, sr=sr)))
        except Exception:
            spec_cent = 0.0

        # MFCC
        try:
            mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
            mfcc_mean = [float(np.mean(mfcc[i])) for i in range(13)]
        except Exception:
            mfcc_mean = [0.0] * 13

        features.append([rms, zcr, spec_cent] + mfcc_mean)

    return features


# ========== PEMROSESAN FOLDER ==========
def process_folder(input_dir, output_csv, sr=16000, frame_length=0.3):
    rows = []

    input_dir = Path(input_dir)
    classes = sorted([p for p in input_dir.iterdir() if p.is_dir()])

    if not classes:
        raise SystemExit("❌ No class folders found. Pastikan struktur dataset benar.")

    supported_ext = ["*.wav", "*.m4a", "*.mp3", "*.ogg", "*.flac"]
    total_files = 0
    success_files = 0
    failed_files = 0

    for cls in classes:
        label = cls.name
        print(f"\n=== Processing class: {label} ===")

        # kumpulkan semua ekstensi
        audio_files = []
        for ext in supported_ext:
            audio_files += list(cls.glob(ext))

        for audio_path in audio_files:
            total_files += 1
            print(f"→ Loading: {audio_path.name}")

            # Coba load langsung via librosa
            try:
                y, sr_use = librosa.load(audio_path, sr=sr)
                load_success = True
            except Exception:
                print("   ⚠ Load failed → trying FFmpeg conversion...")
                load_success = False

            # Jika gagal load → convert via FFmpeg ke WAV temp
            if not load_success:
                temp_wav = audio_path.with_suffix(".temp.wav")
                if convert_to_wav(audio_path, temp_wav):
                    try:
                        y, sr_use = librosa.load(temp_wav, sr=sr)
                        load_success = True
                        temp_wav.unlink(missing_ok=True)  # hapus file sementara
                    except Exception:
                        load_success = False

            if not load_success:
                print("   ❌ FAILED: cannot load even after FFmpeg conversion.")
                failed_files += 1
                continue

            # Skip jika audio terlalu pendek
            if len(y) < sr * 0.2:
                print("   ⚠ Skipped: audio too short.")
                failed_files += 1
                continue

            # Ekstraksi fitur
            feats = extract_features_from_signal(y, sr_use, frame_length_seconds=frame_length)

            if len(feats) == 0:
                print("   ⚠ No valid frames extracted. Skipping.")
                failed_files += 1
                continue

            # Masukkan ke dataframe
            for f in feats:
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "rms": f[0],
                    "zcr": f[1],
                    "spectral_centroid": f[2],
                    "label": label
                }
                for i in range(1, 14):
                    row[f"mfcc_{i}"] = f[2 + i]

                rows.append(row)

            print(f"   ✔ Extracted {len(feats)} frames.")
            success_files += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n==================== RESULTS ====================")
    print(f"Total audio files detected : {total_files}")
    print(f"Successfully processed     : {success_files}")
    print(f"Failed or skipped          : {failed_files}")
    print(f"Total rows in CSV          : {len(df)}")
    print(f"Saved to                   : {output_csv}")
    print("=================================================")


# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust audio feature extractor")
    parser.add_argument("--input_dir", required=True, help="Dataset directory")
    parser.add_argument("--output_csv", default="audio_dataset.csv")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frame_length", type=float, default=0.3)
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_csv, sr=args.sr, frame_length=args.frame_length)
