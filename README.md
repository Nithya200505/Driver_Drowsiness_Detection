\# Driver Drowsiness Detection System



\## Overview



This project is a real-time Driver Drowsiness Detection System developed using Python, OpenCV, and Dlib. It monitors the driver's eyes through a webcam and detects signs of fatigue using the Eye Aspect Ratio (EAR) method. When the driver's eyes remain closed for a certain period, an alarm sound is triggered to alert the driver.



\## Features



\* Real-time face detection

\* Eye landmark detection using Dlib

\* Eye Aspect Ratio (EAR) calculation

\* Continuous alarm when drowsiness is detected

\* Live webcam monitoring

\* Visual warning messages on screen



\## Technologies Used



\* Python

\* OpenCV

\* Dlib

\* SciPy

\* NumPy

\* Imutils



\## Project Structure



```text

Driver\_Drowsiness\_Detection/

│

├── users/

│   └── dlibfatigue/

│       └── Fatigue\_Detection.py

│

├── shape\_predictor\_68\_face\_landmarks.dat

├── requirements.txt

├── README.md

└── .gitignore

```



\## Installation



```bash

git clone https://github.com/Nithya200505/Driver\_Drowsiness\_Detection.git

cd Driver\_Drowsiness\_Detection



pip install -r requirements.txt

```



\## Run the Project



```bash

python users/dlibfatigue/Fatigue\_Detection.py

```



\## Controls



\* Press \*\*Q\*\* to close the application.



\## How It Works



1\. Detects the face using Dlib's frontal face detector.

2\. Extracts facial landmarks using the 68-point landmark model.

3\. Calculates Eye Aspect Ratio (EAR).

4\. Monitors eye closure duration.

5\. Triggers a continuous alarm when drowsiness is detected.



\## Future Improvements



\* Yawn detection

\* Head pose estimation

\* Mobile alerts

\* Deep learning-based fatigue detection

\* Driver analytics dashboard



\## Author



Nithya Sri



