"""
YOLOv8 + face-alignment based proctoring
Features:
1) Eyeball/gaze (left/right/up)
2) Mouth-open detection with baseline
3) Instance/person count (segmentation via YOLO)
4) Phone detection (YOLO)
5) Head-pose estimation (solvePnP)
6) Face spoofing heuristic (blink + motion)
Logs CSV + optional screenshot on CHEATING
"""

import cv2, time, os, math
import numpy as np
import pandas as pd
from ultralytics import YOLO
import face_alignment
from imutils import face_utils

# -------- CONFIG ----------
VIDEO_SOURCE = 0            # 0 = laptop webcam, or "http://IP_ESP32:81/stream"
MODEL_DET = "yolov8n.pt"    # object detection (person, cell phone)
SEG_MODEL = "yolov8n-seg.pt"  # for segmentation (optional, fallback to detection)
CONF_PERSON = 0.45
CONF_PHONE = 0.35

OUTPUT_LOG = "proctor_log.csv"
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Head pose model points (3D model)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin (approx)
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])

# -------- MODELS ----------
print("Loading YOLO models...")
yolo = YOLO(MODEL_DET)      # detection
# for segmentation you can load SEG_MODEL if you want mask outputs:
# yolo_seg = YOLO(SEG_MODEL)

print("Loading face-alignment (landmark detector)...")
import torch

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    flip_input=False
)


# -------- HELPERS ----------
def save_log(row):
    df = pd.DataFrame([row])
    if not os.path.exists(OUTPUT_LOG):
        df.to_csv(OUTPUT_LOG, index=False)
    else:
        df.to_csv(OUTPUT_LOG, mode='a', header=False, index=False)

def mouth_distance(landmarks):
    # use landmarks indices: 62 (upper inner lip), 66 (lower inner lip) for 68-point
    # face_alignment returns 68-point by default
    up = landmarks[62]
    low = landmarks[66]
    return np.linalg.norm(up - low)

def eye_center(landmarks, left=True):
    # left eye: 36-41, right eye: 42-47 (0-based for 68-point)
    if left:
        pts = landmarks[36:42]
    else:
        pts = landmarks[42:48]
    return pts.mean(axis=0)

def compute_gaze(landmarks):
    # approximate gaze by relative iris/eye center shift
    # we compute left and right eye centers and nose x; compare centers relative to eye corners
    left_center = eye_center(landmarks, left=True)
    right_center = eye_center(landmarks, left=False)
    nose = landmarks[30]  # nose tip
    # average eye x vs nose x
    eyes_x = (left_center[0] + right_center[0]) / 2.0
    dx = nose[0] - eyes_x
    # threshold tuned empirically (may need calibration)
    if dx > 6:        # nose shifted right -> looking left (camera perspective)
        return "LEFT"
    if dx < -6:
        return "RIGHT"
    # check vertical: use eye center y vs nose y
    eyes_y = (left_center[1] + right_center[1]) / 2.0
    dy = eyes_y - nose[1]
    if dy < -6:
        return "UP"
    return "CENTER"

def get_head_pose(landmarks, img_w, img_h):
    # select image_points corresponding to MODEL_POINTS roughly:
    # nose tip 30, chin 8, left eye corner 36, right eye corner 45, left mouth 48, right mouth 54
    img_pts = np.array([
        landmarks[30],  # nose tip
        landmarks[8],   # chin
        landmarks[36],  # left eye left corner
        landmarks[45],  # right eye right corner
        landmarks[48],  # left mouth
        landmarks[54],  # right mouth
    ], dtype="double")

    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))
    success, rotation_vec, translation_vec = cv2.solvePnP(MODEL_POINTS, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
    x = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
    y = math.degrees(math.atan2(-rmat[2,0], sy))
    z = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
    # return pitch (x), yaw (y), roll (z)
    return x, y, z

# -------- STATE ----------
MOUTH_BASELINE = None
prev_face = None
blink_timestamps = []
last_status = None
cheat_start = None

# -------- MAIN LOOP ----------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Cannot open camera/source:", VIDEO_SOURCE)
    raise SystemExit

print("Starting main loop. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    status = "NORMAL"
    timestamp = time.time()

    # 1) YOLO detection (persons, phones)
    results = yolo(frame)[0]  # non-stream simple usage
    person_count = 0
    phone_detected = False

    # boxes: results.boxes
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = yolo.model.names[cls] if hasattr(yolo, "model") else yolo.names.get(cls, str(cls))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if label == "person" and conf >= CONF_PERSON:
            person_count += 1
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        if label in ["cell phone", "mobile phone", "phone"] and conf >= CONF_PHONE:
            phone_detected = True
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"PHONE",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    # segmentation: if you loaded seg model, you can get masks and count people by mask area.
    # fallback: person_count from detection (above)

    # 2) Face landmarks (if at least one person)
    face_landmarks = None
    # crop a face region for speed: choose largest person bbox if exists
    if person_count >= 1:
        # pick largest bbox from results
        largest = None; max_area = 0
        for box in results.boxes:
            cls = int(box.cls[0])
            label = yolo.model.names[cls] if hasattr(yolo, "model") else yolo.names.get(cls, str(cls))
            if label != "person": continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2-x1)*(y2-y1)
            if area > max_area:
                max_area = area; largest = (x1,y1,x2,y2)
        if largest is not None:
            x1,y1,x2,y2 = largest
            face_img = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            try:
                # face_alignment expects RGB image
                fa_arr = face_alignment.utils.get_align_face(face_img) if False else None
            except Exception:
                fa_arr = None
            # run face_alignment on whole frame (slower) if crop fails
            try:
                preds = fa.get_landmarks(frame)
            except Exception:
                preds = None
            if preds and len(preds) > 0:
                # use first detected face
                landmarks = preds[0]  # (68,2)
                face_landmarks = landmarks

    # 3) If we have landmarks -> gaze, mouth, head pose, blink
    if face_landmarks is not None:
        # mouth open
        m_dist = mouth_distance(face_landmarks)
        if MOUTH_BASELINE is None:
            MOUTH_BASELINE = m_dist
        if m_dist > MOUTH_BASELINE * 1.8:
            status = "SUSPECT (MOUTH OPEN)"

        # gaze
        gaze = compute_gaze(face_landmarks)
        if gaze in ["LEFT","RIGHT","UP"]:
            # if gaze away for a short time, label suspect; prolonged -> cheating
            status = "SUSPECT (GAZE:"+gaze+")"

        # head pose
        pitch, yaw, roll = get_head_pose(face_landmarks, w, h)
        # yaw > 25 deg => suspect, pitch > 25 deg for prolonged -> cheating
        if abs(yaw) > 25:
            status = "SUSPECT (HEAD TURN)"
        if pitch > 25:
            if cheat_start is None:
                cheat_start = timestamp
            elif timestamp - cheat_start > 3.0:
                status = "CHEATING (LOOK DOWN)"
        else:
            cheat_start = None

        # blink detection (simple): measure eye aspect ratio from landmarks
        # left eye pts 36-41, right 42-47
        left_ear = np.linalg.norm(face_landmarks[37]-face_landmarks[41]) / np.linalg.norm(face_landmarks[36]-face_landmarks[39] + 1e-8)
        right_ear = np.linalg.norm(face_landmarks[43]-face_landmarks[47]) / np.linalg.norm(face_landmarks[42]-face_landmarks[45] + 1e-8)
        ear = (left_ear + right_ear) / 2.0
        # heuristic: very low ear across many frames -> spoof (no micro-movements)
        if ear < 0.15:
            blink_timestamps.append(timestamp)

        # detect face motion variance
        if prev_face is None:
            prev_face = face_landmarks.copy()
            motion_var = 0.0
        else:
            motion_var = np.mean(np.linalg.norm(face_landmarks - prev_face, axis=1))
            prev_face = face_landmarks.copy()
        if motion_var < 0.2 and len(blink_timestamps) < 2:
            # possible spoof (static photo)
            status = "CHEATING (SPOOF SUSPECT)"

    # 4) People/phone rules override
    if person_count == 0:
        status = "SUSPECT (NO PERSON)"
    elif person_count > 1:
        status = "CHEATING (MULTIPLE PERSON)"
    if phone_detected:
        status = "CHEATING (PHONE)"

    # 5) Display & logging
    cv2.putText(frame, f"Status: {status}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255) if "CHEATING" in status else (0,255,255), 2)
    cv2.putText(frame, f"Persons: {person_count}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    if face_landmarks is not None:
        cv2.putText(frame, f"Gaze: {gaze}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    cv2.imshow("PROCTORING (YOLO + Landmarks)", frame)

    # take screenshot and log if cheating
    if "CHEATING" in status:
        fname = os.path.join(SCREENSHOT_DIR, f"{int(time.time())}.jpg")
        cv2.imwrite(fname, frame)
        log_row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "persons": person_count, "phone": phone_detected, "gaze": (gaze if 'gaze' in locals() else ""), "motion_var": float(motion_var if 'motion_var' in locals() else 0.0)}
        save_log(log_row)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
