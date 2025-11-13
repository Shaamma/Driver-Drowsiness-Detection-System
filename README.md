# Driver Drowsiness Detection System

A deep learning–based driver safety project that detects whether a driver’s eyes are open or closed using a Convolutional Neural Network (CNN). The model is integrated into a real-time webcam system that monitors eye closure duration and triggers a drowsiness alert when eyes remain closed for more than 3 seconds.

---

## Features

- **CNN-based Eye State Classification**  
  Classifies eye images into Open or Closed using a custom-trained deep learning model.

- **Complete ML Pipeline**  
  Includes dataset cleaning, augmentation, class-imbalance handling, model training, and evaluation.

- **Real-Time Detection with OpenCV**  
  Detects the face and eyes from webcam feed using Haar Cascades.

- **Drowsiness Alert System**  
  Triggers an on-screen alert when eyes stay closed continuously for 3 seconds.

---

## Dataset Structure

Structure your dataset as follows:
data/
├── train/
│ ├── Open_Eyes/
│ └── Closed_Eyes/
└── test/
├── Open_Eyes/
└── Closed_Eyes/


Each folder contains eye images labeled according to the state.

---

## Model Training

The model training script:

- Loads and cleans dataset  
- Splits data into train and validation sets  
- Applies augmentation and class weighting  
- Builds a CNN using TensorFlow/Keras  
- Saves the trained model to:

models/best_drowsiness_model.h5


### Run Training
python train_drowsiness_model.py


---

## Real-Time Drowsiness Detection

The real-time system:

- Loads the trained CNN model  
- Detects face and eyes in video feed  
- Classifies eye state frame-by-frame  
- Tracks how long eyes stay closed  
- Shows alert after 3 consecutive seconds of closure  

### Run Real-Time System

python realtime_drowsiness_alert.py


Press **q** to exit.

---

## Model Evaluation

The evaluation includes:

- Accuracy score  
- Loss curves  
- Confusion matrix  
- Classification report  

These results help validate model performance and identify misclassifications.

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas, Scikit-learn  
- Matplotlib / Seaborn  

---

## How It Works

1. CNN predicts eye state (Open / Closed) for every detected eye.  
2. Timer starts when eyes first appear Closed.  
3. If eyes remain Closed for 3 seconds → Drowsiness Alert is triggered.  
4. Timer resets when eyes open again.  
5. The system runs continuously using the webcam feed.

---

## Future Improvements

- Add yawning detection  
- Use facial landmark tracking instead of Haar Cascades  
- Add audio alert system  
- Deploy as mobile or dashboard application  

---

## Author

**Shaamma**  
Driver Drowsiness Detection Project  
GitHub: https://github.com/Shaamma

---

## Acknowledgements

- OpenCV Haar Cascades  
- TensorFlow & Keras  
- Dataset contributors (Open_Eyes / Closed_Eyes)


