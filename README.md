# 🚗 Driver Drowsiness Detection System

A deep learning–based Driver Drowsiness Detection system that identifies whether a driver’s eyes are **Open** or **Closed** using a Convolutional Neural Network (CNN). The trained model is integrated with a real-time webcam application that triggers an alert when the driver’s eyes remain closed for more than **3 seconds**, indicating possible drowsiness.

---

## 📌 Features

- **CNN-based Eye State Classification**  
  Classifies eye images into *Open* or *Closed* with a custom-trained model.

- **Complete ML Pipeline**  
  Includes dataset cleaning, augmentation, class-imbalance handling, model training, and evaluation.

- **Real-Time Detection**  
  Uses OpenCV and Haar Cascades to detect eyes from webcam feed and track eye closure duration.

- **Drowsiness Alert**  
  If eyes stay closed for **≥ 3 seconds**, the system displays a warning alert on-screen.

---

## 📁 Dataset Structure

Place your dataset inside:

data/
├── train/
│ ├── Open_Eyes/
│ └── Closed_Eyes/
└── test/
├── Open_Eyes/
└── Closed_Eyes/


---

## 🧠 Model Training

The training script:

- Loads and cleans dataset  
- Splits training data into train/validation  
- Applies augmentation and class weighting  
- Builds and trains a CNN  
- Saves the best model to:  

models/best_drowsiness_model.h5


### ▶️ Run Training

```bash
python train_drowsiness_model.py
```

🎥 Real-Time Detection

The real-time script:

Loads the trained CNN model

Uses OpenCV to detect face and eyes

Classifies eye state per frame

Tracks how long eyes remain closed

Triggers alert if closed for 3 seconds

▶️ Run Real-Time System
python realtime_drowsiness_alert.py
Press q to quit.

🧪 Model Evaluation

The evaluation includes:

Accuracy

Loss curves

Confusion matrix

Classification report

These help validate model performance and identify misclassification patterns.

🛠️ Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy, Pandas, Scikit-learn

Matplotlib / Seaborn

Haar Cascades (face & eye detection)

🚀 How It Works

CNN predicts whether each detected eye is Open or Closed

System starts a timer when eyes first appear Closed

If eyes remain closed for ≥ 3 seconds → Drowsiness Alert

Timer resets when eyes open again

Runs continuously through webcam feed

📊 Results

Reliable classification of Open vs Closed eyes

Smooth real-time performance

Accurate drowsiness detection using time-based logic

Extendable to include yawning, head pose, or PERCLOS calculation

📌 Future Improvements

Add yawning detection

Use facial landmarks instead of Haar cascades

Deploy as a mobile or dashboard application

Integrate audio alerts or IoT-based warnings

🧑‍💻 Author

Shaamma
Driver Drowsiness Detection
GitHub: https://github.com/Shaamma

⭐ Acknowledgements

OpenCV Haar Cascades

TensorFlow & Keras

Dataset contributors (Open_Eyes / Closed_Eyes dataset)

If this helped you, consider giving the repo a ⭐!

