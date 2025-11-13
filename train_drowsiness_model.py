import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, UnidentifiedImageError

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------
# CONFIG
# --------------------
sns.set(style="whitegrid")

BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

IMG_SIZE   = (80, 80)
BATCH_SIZE = 32
EPOCHS     = 25
RANDOM_SEED = 42

os.makedirs("models", exist_ok=True)

# --------------------
# 1. CLEANING: REMOVE CORRUPT IMAGES
# --------------------
def is_image_ok(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def clean_directory(root_dir, exts=(".jpg", ".jpeg", ".png")):
    removed = 0
    for root, _, files in os.walk(root_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue
            full_path = os.path.join(root, f)
            if not is_image_ok(full_path):
                print("Removing corrupt image:", full_path)
                try:
                    os.remove(full_path)
                    removed += 1
                except OSError:
                    pass
    print(f"[CLEAN] Removed {removed} corrupt images from {root_dir}")

clean_directory(TRAIN_DIR)
clean_directory(TEST_DIR)

# --------------------
# 2. DATA GENERATORS (train/val from train/, test from test/)
# --------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2    # 20% of train used as validation
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=RANDOM_SEED
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=RANDOM_SEED
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("[INFO] Class indices:", train_gen.class_indices)
# Expect something like: {'Closed_Eyes': 0, 'Open_Eyes': 1}

# --------------------
# 3. SIMPLE EDA – CLASS DISTRIBUTION & SAMPLE IMAGES
# --------------------
def plot_class_distribution(generator, title):
    labels = generator.classes
    idx_to_class = {v: k for k, v in generator.class_indices.items()}

    plt.figure(figsize=(5,4))
    sns.countplot(x=labels)
    plt.xticks([0, 1], [idx_to_class[0], idx_to_class[1]])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

plot_class_distribution(train_gen, "Train Class Distribution")
plot_class_distribution(val_gen, "Validation Class Distribution")
plot_class_distribution(test_gen, "Test Class Distribution")

# Show a few sample images
x_batch, y_batch = next(train_gen)
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = x_batch[i].reshape(IMG_SIZE[0], IMG_SIZE[1])
    plt.imshow(img, cmap="gray")
    label = idx_to_class[int(y_batch[i])]
    plt.title(label)
    plt.axis("off")
plt.tight_layout()
plt.show()

# --------------------
# 4. CLASS WEIGHTS (handle imbalance)
# --------------------
y_train = train_gen.classes
class_labels = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=class_labels,
    y=y_train
)
class_weights = {int(i): float(class_weights_array[i]) for i in range(len(class_labels))}
print("[INFO] Class weights:", class_weights)

# --------------------
# 5. BUILD CNN MODEL
# --------------------
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Closed_Eyes vs Open_Eyes
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------
# 6. TRAIN WITH EARLY STOPPING
# --------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "models/best_drowsiness_model.h5",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights
)

# --------------------
# 7. TRAINING CURVES
# --------------------
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------
# 8. TEST EVALUATION
# --------------------
test_loss, test_acc = model.evaluate(test_gen)
print(f"[RESULT] Test Loss: {test_loss:.4f}")
print(f"[RESULT] Test Accuracy: {test_acc:.4f}")

# Predictions
y_prob = model.predict(test_gen)
y_pred = (y_prob > 0.5).astype(int).ravel()
y_true = test_gen.classes

idx_to_class_test = {v: k for k, v in test_gen.class_indices.items()}
target_names = [idx_to_class_test[0], idx_to_class_test[1]]

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(4,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=target_names,
    yticklabels=target_names,
    cbar=False
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
