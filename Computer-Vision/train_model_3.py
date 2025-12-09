import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import cv2
import matplotlib.pyplot as plt

# =====================================================================
# ✅ CONFIG
# =====================================================================
IMAGE_DIR = "dataset_photo/Dataset"
CSV_FILE  = "dataset_photo/auto_labeled_dataset.csv"
OUTPUT_DIR = "Computer-Vision/models_output"

BATCH_SIZE = 16
EPOCHS = 10
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# ✅ 1. LOAD CSV
# =====================================================================
print("📥 Loading CSV...")
df = pd.read_csv(CSV_FILE)
print(df.head())
print(df.isnull().sum())
print(df["label"].value_counts())
print("Total data:", len(df))

# =====================================================================
# ✅ 2. CLEAN DATA
# =====================================================================
print("🧹 Cleaning data...")
df = df.dropna()
df = df[df["image"].apply(lambda x: os.path.exists(os.path.join(IMAGE_DIR, x)))]
print("Clean data:", len(df))

# =====================================================================
# ✅ 3. LABEL ENCODING
# =====================================================================
print("🏷️ Encoding labels...")

le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
NUM_CLASSES = len(le.classes_)
print("Classes:", le.classes_)

encoder_path = os.path.join(OUTPUT_DIR, "vision_label_encoder.joblib")
joblib.dump(le, encoder_path)
print("Saved LabelEncoder →", encoder_path)

# =====================================================================
# ✅ 4. TRAIN VAL TEST SPLIT (70 - 15 - 15)
# =====================================================================
print("🔀 Splitting dataset 70/15/15...")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label_encoded"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label_encoded"]
)

print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

# =====================================================================
# ✅ 5. CUSTOM DATASET
# =====================================================================
class CheatingDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image"]
        label    = self.df.loc[idx, "label_encoded"]

        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image not found or unreadable: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================================
# ✅ 6. TRANSFORMATION & DATALOADER
# =====================================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

train_dataset = CheatingDataset(train_df, IMAGE_DIR, transform)
val_dataset   = CheatingDataset(val_df,   IMAGE_DIR, transform)
test_dataset  = CheatingDataset(test_df,  IMAGE_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# =====================================================================
# ✅ 7. MODEL (RESNET18 TRANSFER LEARNING)
# =====================================================================
print("🧠 Building model...")

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =====================================================================
# ✅ 8. TRAINING + VALIDATION
# =====================================================================
print("🚀 Training started...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # ================== VALIDATION ==================
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {total_loss:.3f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%")

# =====================================================================
# ✅ 9. TEST EVALUATION
# =====================================================================
print("\n📊 Evaluation on TEST set...")

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

# =====================================================================
# ✅ 10. SAVE MODEL
# =====================================================================
model_path = os.path.join(OUTPUT_DIR, "cheating_cnn_model.pth")
torch.save(model.state_dict(), model_path)

print("\n✅ TRAINING SELESAI")
print("✅ Model   :", model_path)
print("✅ Encoder :", encoder_path)
