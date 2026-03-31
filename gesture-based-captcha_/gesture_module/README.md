# 🚀 Gesture-Based CAPTCHA - Pro Edition 🔥

## Structure
```
gesture_module/
├── dataset/      (collect images here)
├── collected_data/
├── models/gesture_model.h5
├── utils/           (hand_detector.py, preprocessing.py)
├── data_collection.py  🔥
├── train_model.py     🔥 (enhanced CNN)
├── predict_gesture.py 🔥 (hand detect + predict)
├── captcha_engine.py
├── main_gesture.py    🔥 (full CAPTCHA)
├── requirements.txt
└── README.md
```

## 🎯 How to Run (Step-by-Step)

### 1. Setup (Virtual Env Recommended)
```
cd gesture_module
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Collect Data (2-3 min/gesture)
```
python data_collection.py
```
- Input: `Fist` → Hold fist in **green box** → Auto-save every 10s
- Repeat for: Fist, One, Two, Palm, ThumbsUp, OK
- **Pro Tip:** 100-200 images each, vary angles/lighting → `collected_data/` or `dataset/`

### 3. Train Model (~5-10 min)
```
python train_model.py
```
- Uses preprocessing, validation split, dropout for better accuracy
- Saves `models/gesture_model.h5`

### 4. Test Prediction
```
python predict_gesture.py
```
- **Green box ROI** + MediaPipe hand detection
- Shows gesture + confidence %
- Press `Q` to quit

### 5. Run CAPTCHA Demo 🔥
```
python main_gesture.py
```
- Generates random gesture challenge
- Detects + verifies with 80% confidence threshold
- Complete 3 correct gestures to pass

## Improvements Made ✅
- ✅ **Utils Integration:** preprocessing (grayscale+normalize), hand_detector (MediaPipe)
- ✅ **Enhanced Training:** Error handling, filters, val split, dropout, 15 epochs
- ✅ **Better Prediction:** Hand detect gate, confidence display, Q quit
- ✅ **Full CAPTCHA:** Multi-challenge, stable detection, visual feedback
- ✅ **Robust:** Handles missing imgs, grayscale→RGB adapt

## Troubleshooting
| Issue | Fix |
|-------|-----|
| No webcam | Check `cv2.VideoCapture(0)` index |
| Low accuracy | More data + train longer |
| Import error | `pip install -r requirements.txt` |
| Hand not detected | Adjust lighting/hand position |

## Pro Tips 💎
```
# Faster collection (1s interval): Edit data_collection.py >10 to >1
if current_time - last_save_time > 1:
```
- **Lighting:** Natural + lamp variations
- **Angles:** Straight, tilted 30-60°
- **Extend:** Add `run_all.py` for batch training

## 🎉 New Improvements (BLACKBOXAI)

### Key Enhancements:
- **config.py**: Centralized settings (paths, thresholds, timeouts)
- **Dynamic ROI**: Auto-adjusts green box to detected hand bbox + padding
- **Advanced CAPTCHA**: Multi-gesture sequences (Fist → Palm), adaptive difficulty, 10s timeouts
- **Better UX**: Confidence-based prediction, low-conf warning, session progress, remaining time countdown, instructions
- **Robust**: Model/cam error handling, conf threshold gate, safe attr access
- **Performance**: Global MediaPipe (no per-frame init), 96x96 preprocess match training
- **utils/ui_helpers.py**: Reusable UI components (progress bars, etc.)

### Usage:
```
cd gesture_module
python main_gesture.py  # Enhanced CAPTCHA demo
```

✅ **main_gesture.py fully improved + project polished!**

**Ready to deploy! Webcam → Gesture → CAPTCHA solved 😎**

