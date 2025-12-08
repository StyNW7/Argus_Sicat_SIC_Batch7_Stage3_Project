import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
import time
from sklearn.preprocessing import LabelEncoder
import torch.serialization

# ================== CONFIG ==================
MODEL_PATH = "cheating_cnn_model.pth"
ENCODER_PATH = "label_encoder.pt"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== LOAD LABEL ENCODER (FIX PYTORCH 2.6+) ==================
print("🔠 Loading label encoder...")
torch.serialization.add_safe_globals([LabelEncoder])
le = torch.load(ENCODER_PATH, weights_only=False)

# ================== LOAD MODEL ==================
print("🧠 Loading CNN model...")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(le.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ================== TRANSFORM ==================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ================== WEBCAM ==================
cap = cv2.VideoCapture(0)
print("📷 Webcam started... Press Q to exit")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam error")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(DEVICE)

    # ================== PREDICTION ==================
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, pred_class = torch.max(probs, 1)

    label = le.inverse_transform([pred_class.item()])[0]
    confidence = conf.item() * 100

    # ================== FPS ==================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # ================== DISPLAY ==================
    color = (0,255,0)

    if label == "cheating":
        color = (0,0,255)
    elif label == "suspect":
        color = (0,255,255)

    text = f"{label.upper()} ({confidence:.1f}%) | FPS: {fps:.1f}"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Cheating Detection - CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
