# Face + Gesture 2FA Authentication System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-blue.svg)](https://mediapipe.dev/)

A **dual-factor authentication (2FA) system** combining **facial recognition** and **gesture-based CAPTCHA** for secure, liveness-detected access control. Built for anti-spoofing with real-time face re-verification during gesture challenges.

## 🎯 Features

- **Face Recognition**: Uses `face_recognition` library for accurate identity verification (tolerance 0.45)
- **Gesture CAPTCHA**: 6-class hand gesture recognition (Fist, One, Two, Palm, ThumbsUp, OK)
  - MediaPipe hand detection + CNN model (96x96 grayscale)
  - Multi-gesture sequences (configurable: 3 correct required)
  - Hold detection (2s minimum) + 80% confidence threshold
  - 15s per-gesture timeout, 3 retries max
- **Anti-Spoofing**: Continuous face re-verification every 30 frames during gestures
- **Production-Ready**: Logging (`auth.log`), error handling, FPS monitoring, robust state machine
- **Visual Feedback**: Real-time UI with confidence %, timers, progress, status

## 🏗️ Project Structure

```
fulll_minor_project/
├── main_2.py                 # Main 2FA application (Face → Gesture)
├── gesture-based-captcha_/   # Gesture CAPTCHA module
│   ├── gesture_module/
│   │   ├── dataset/          # Training images (Fist/, One/, etc. ~150/img)
│   │   ├── utils/            # hand_detector.py, preprocessing.py, ui_helpers.py
│   │   ├── config.py         # Centralized settings (paths, thresholds)
│   │   ├── data_collection.py # Webcam data capture
│   │   ├── train_model.py    # CNN training (15 epochs, dropout, val_split)
│   │   ├── predict_gesture.py # Standalone prediction test
│   │   ├── main_gesture.py   # Gesture CAPTCHA demo
│   │   ├── captcha_engine.py # Core verification logic
│   │   └── requirements.txt
│   └── TODO.md
├── face_module/              # Face recognition pipeline
│   ├── dataset_scripts/      # extract_faces.py, encode_faces.py, etc.
│   ├── dataset/              # Face images + embeddings/
│   └── pipeline/             # face_pipeline.py
├── auth.log                  # Authentication logs
├── .gitignore
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Recommended: Python 3.8+
cd c:/Users/Lenovo/fulll_minor_project
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r gesture-based-captcha_/gesture_module/requirements.txt
pip install face_recognition  # Additional for faces
```

### 2. Prepare Face Database
```bash
cd face_module/dataset_scripts
python extract_faces.py    # Extract from videos/images
python encode_faces.py     # Generate embeddings/encodings.npy + names.npy
```

### 3. Train Gesture Model (if needed)
```bash
cd gesture-based-captcha_/gesture_module
python data_collection.py  # Collect more data if needed
python train_model.py      # ~5-10 min, outputs models/gesture_model.h5
```

### 4. Run Full 2FA System
```bash
python main_2.py
```
- **Step 1**: Face detected → verified
- **Step 2**: Gesture challenges (random sequence)
- **Step 3**: SUCCESS or ACCESS DENIED

**Controls**: `Q` or `ESC` to quit

## 📱 Demo Flow

```
[Camera] → Face Recognition → Gesture Sequence (3x)
          ↓
     "Show: Fist" (15s, hold 2s)
     "✓ Holding 1.2s..." → Next: "Palm"
          ↓
       "🎉 SUCCESS!" (logged)
```

## 🔧 Configuration

Edit `gesture-based-captcha_/gesture_module/config.py`:
```python
NUM_CORRECT_REQUIRED = 3
TIMEOUT_SEC = 15
HOLD_REQUIRED = 2.0
CONFIDENCE_THRESHOLD = 80  # %
ROI_PADDING = 20
```

## 🛠️ Key Components

| Module | Purpose | Key Tech |
|--------|---------|----------|
| `main_2.py` | Full 2FA orchestration | State machine, logging |
| `hand_detector.py` | Hand ROI detection | MediaPipe Hands |
| `train_model.py` | Gesture CNN training | TF 2.15, Dropout, ValSplit |
| `face_recognition` | Identity verification | dlib HoG + 128D encodings |

## 📈 Performance

- **FPS**: 25-30 (optimized, buffer=1, frame skip)
- **Accuracy**: 90%+ gestures (varied dataset)
- **Face Tolerance**: 0.45 (robust to lighting/angle)
- **Gestures**: Fist(150+ imgs), One, Two, Palm, ThumbsUp, OK

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Camera failed" | Try `cv2.VideoCapture(1)` or check privacy settings |
| Low gesture acc | Collect 200+/gesture, retrain |
| "No face_locations" | Good lighting, face 50-80cm away |
| Import errors | `pip install -r requirements.txt` + `face_recognition` |
| Model not found | Run `train_model.py` first |

## 📋 TODO (from project files)

- [x] Gesture training path fixes
- [ ] Test full pipeline end-to-end
- [ ] Add more gestures/dataset variety
- [ ] Deploy as web service (Flask/FastAPI)

## 🤝 Dependencies

```txt
opencv-python
tensorflow==2.15.0
numpy
mediapipe
scikit-learn==1.5.1
face_recognition  # pip install face_recognition
```

