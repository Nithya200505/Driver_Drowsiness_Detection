Driver Drowsiness Classification using Eye Blink and Head Movement (KNN)

Overview
This project aims to detect driver drowsiness by analyzing eye blink rate and head movement patterns using the K-Nearest Neighbors (KNN) algorithm. It can help prevent accidents by alerting drowsy drivers in real-time.

Features
Real-time eye blink detection
Head movement analysis
KNN-based classification for drowsiness detection
Alerts when drowsiness is detected

Project Structure
driver-drowsiness-detection/
│
├── dataset/                 # Collected data (eye blink, head position)
├── src/                     # Main source code
│   ├── eye_blink_detector.py
│   ├── head_movement_tracker.py
│   └── knn_classifier.py
├── model/                   # Saved models
├── utils/                   # Helper scripts
├── README.md
└── requirements.txt

Installation
git clone https://github.com/vamshiambatipudi/driver-drowsiness-detection.git
cd driver-drowsiness-detection
pip install -r requirements.txt

Algorithm
Machine Learning Algorithm: K-Nearest Neighbors (KNN)

Features used: blink rate, head tilt, frequency of nodding, etc.

Output: Open - Score/ Close - Score

Requirements
Python 3.x
OpenCV
dlib or mediapipe (for facial landmarks)
scikit-learn
numpy, matplotlib
