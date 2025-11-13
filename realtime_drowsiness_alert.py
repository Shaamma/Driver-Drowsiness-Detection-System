import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "models/best_drowsiness_model.h5"
IMG_SIZE = (80, 80)

ALERT_SECONDS = 3.0  # trigger if eyes closed this long

# Load model
model = load_model(MODEL_PATH)
print("[INFO] Loaded model from", MODEL_PATH)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier("haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    print("[ERROR] Could not load Haar cascades.")
    exit()

# --------------------
# HELPERS
# --------------------
def preprocess_eye(eye_img):
    """Convert BGR eye patch to CNN input."""
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    norm = resized.astype("float32") / 255.0
    return np.expand_dims(norm, axis=(0, -1))  # (1, H, W, 1)

def predict_eye_state(eye_img):
    """
    Returns ("Open" or "Closed", prob_open) for a given BGR eye patch.
    prob_open ~ 1 means open, ~0 means closed.
    """
    eye_input = preprocess_eye(eye_img)
    prob_open = model.predict(eye_input, verbose=0)[0][0]
    state = "Open" if prob_open >= 0.5 else "Closed"
    return state, prob_open

# --------------------
# REAL-TIME LOOP
# --------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print(f"[INFO] Drowsiness alert if eyes closed continuously for {ALERT_SECONDS} seconds.")
print("[INFO] Press 'q' to quit.")

closed_start_time = None      # when eyes first detected as closed
alert_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    frame_state = "Open"   # assume open, change if we see a closed eye
    frame_prob = 0.0
    any_eye_detected = False

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        roi_gray  = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in face ROI
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            any_eye_detected = True
            eye_patch = roi_color[ey:ey+eh, ex:ex+ew]

            state, prob_open = predict_eye_state(eye_patch)
            frame_state = state
            frame_prob = prob_open

            color = (0, 255, 0) if state == "Open" else (0, 0, 255)

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(roi_color, f"{state} ({prob_open:.2f})",
                        (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # We just use the first detected eye for timing logic
            break

        # Only handle first face for simplicity
        break

    # --------------------
    # TIME-BASED DROWSINESS LOGIC
    # --------------------
    current_time = time.time()

    if any_eye_detected and frame_state == "Closed":
        if closed_start_time is None:
            closed_start_time = current_time  # just started closing
        else:
            elapsed = current_time - closed_start_time
            if elapsed >= ALERT_SECONDS:
                alert_triggered = True
    else:
        # Eyes not closed right now -> reset timer
        closed_start_time = None
        alert_triggered = False

    # Overlay status text
    if closed_start_time is not None:
        elapsed = current_time - closed_start_time
    else:
        elapsed = 0.0

    status_text = f"Eye: {frame_state}  Closed for: {elapsed:.1f}s"
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if alert_triggered:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Exited.")
