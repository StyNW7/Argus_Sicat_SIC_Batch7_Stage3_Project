#!/usr/bin/env python3
"""
generate_dataset_no_yolo.py

Features:
- Tanpa YOLO (NO object detection)
- Menggunakan face_alignment untuk:
  - gaze
  - mouth open
  - head pose (yaw & pitch)
- Robust terhadap landmark gagal / gambar rusak
- Output CSV + auto label:
  CHEATING / SUSPECT / NOT_CHEATING
"""

import os
import math
from pathlib import Path
from datetime import datetime
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import face_alignment

# ---------------- CONFIG ----------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_IMAGE_DIR = "../Dataset"
OUTPUT_CSV_DEFAULT = "auto_labeled_dataset.csv"

RESIZE_TO = (640, 480)

# ---------------- LOAD MODELS ----------------
def load_models():
    print("Loading face-alignment...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        flip_input=False
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    return fa, face_cascade

# ---------------- HELPERS ----------------
def mouth_distance(lm):
    try:
        return float(np.linalg.norm(lm[62] - lm[66]))
    except:
        return 0.0

def eye_center(lm, left=True):
    if left:
        return lm[36:42].mean(axis=0)
    return lm[42:48].mean(axis=0)

def compute_gaze(lm):
    try:
        left_center = eye_center(lm, True)
        right_center = eye_center(lm, False)
        nose = lm[30]

        eyes_x = (left_center[0] + right_center[0]) / 2.0
        dx = nose[0] - eyes_x

        if dx > 6: return "LEFT"
        if dx < -6: return "RIGHT"

        eyes_y = (left_center[1] + right_center[1]) / 2.0
        dy = eyes_y - nose[1]
        if dy < -6: return "UP"
    except:
        pass

    return "CENTER"

def safe_get_head_pose(lm, w, h):
    try:
        if lm is None or len(lm) < 68:
            return None, None

        required_idx = [30, 8, 36, 45, 48, 54]
        img_pts = np.array([lm[i] for i in required_idx], dtype=np.float64)

        if img_pts.shape != (6, 2):
            return None, None

        model_pts = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        focal = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal, 0.0, center[0]],
            [0.0, focal, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            model_pts, img_pts,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)

        pitch = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
        yaw = math.degrees(math.atan2(-rmat[2,0], sy))

        return pitch, yaw
    except:
        return None, None

# ---------------- LABEL LOGIC (SEIMBANG) ----------------
def classify_label(gaze, yaw, pitch, has_face):

    # Tanpa wajah = mencurigakan
    if not has_face:
        return "SUSPECT"

    # Aman / normal
    if gaze == "CENTER" and abs(yaw) < 15 and abs(pitch) < 15:
        return "NOT_CHEATING"

    # Mencurigakan ringan
    if gaze in ["LEFT", "RIGHT"] or abs(yaw) >= 15 or abs(pitch) >= 15:
        return "SUSPECT"

    return "NOT_CHEATING"

# ---------------- PROCESS IMAGE ----------------
def process_image(path, fa, face_cascade):
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None, "corrupt"

        img = cv2.resize(img, RESIZE_TO)
        h, w = img.shape[:2]

        has_face = 0
        mouth_open = 0.0
        gaze = "CENTER"
        pitch_val = None
        yaw_val = None

        try:
            preds = fa.get_landmarks(img)
        except:
            preds = None

        if preds is not None and len(preds) > 0:
            lm = preds[0]
            if lm is not None and len(lm) >= 68:
                has_face = 1
                mouth_open = mouth_distance(lm)
                gaze = compute_gaze(lm)
                pitch_val, yaw_val = safe_get_head_pose(lm, w, h)

        pitch_csv = float(pitch_val) if pitch_val is not None else 0.0
        yaw_csv = float(yaw_val) if yaw_val is not None else 0.0

        label = classify_label(gaze, pitch_csv, yaw_csv, bool(has_face))

        row = {
            "timestamp": datetime.now().isoformat(),
            "image": path.name,
            "yaw": yaw_csv,
            "pitch": pitch_csv,
            "gaze": gaze,
            "mouth_open": mouth_open,
            "has_face": int(has_face),
            "label": label
        }

        return row, "ok"

    except Exception as e:
        return None, f"error:{e}"

# ---------------- PROCESS FOLDER ----------------
def process_folder(input_dir, output_csv):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    fa, face_cascade = load_models()

    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    if not image_files:
        raise SystemExit("❌ No images found in dataset folder.")

    rows = []
    total = 0
    ok_cnt = 0
    fail_cnt = 0

    for img_path in sorted(image_files):
        total += 1
        print(f"[{total}] {img_path.name} ...", end=" ", flush=True)

        row, status = process_image(img_path, fa, face_cascade)

        if status == "ok":
            rows.append(row)
            ok_cnt += 1
            print("OK")
        else:
            fail_cnt += 1
            print("FAIL")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n================ RESULTS ================")
    print("Total images scanned :", total)
    print("Successfully parsed  :", ok_cnt)
    print("Failed images        :", fail_cnt)
    print("Rows saved to CSV    :", len(df))
    print("CSV path             :", output_csv)
    print("========================================")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset (NO YOLO)")
    parser.add_argument("--input_dir", "-i", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output_csv", "-o", type=str, default=OUTPUT_CSV_DEFAULT)
    parser.add_argument("--resize_w", type=int, default=RESIZE_TO[0])
    parser.add_argument("--resize_h", type=int, default=RESIZE_TO[1])
    args = parser.parse_args()

    RESIZE_TO = (args.resize_w, args.resize_h)

    process_folder(args.input_dir, args.output_csv)
