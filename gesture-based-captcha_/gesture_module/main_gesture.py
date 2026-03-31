import cv2
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.hand_detector import detect_hand
from utils.preprocessing import preprocess
from utils.ui_helpers import draw_exact_ui
from config import *

# Config - using imported from config.py
model = load_model(MODEL_PATH)
# Local overrides if needed:
TIMEOUT_SEC = 15.0  # longer for testing
NUM_SUCCESS = NUM_CORRECT_REQUIRED

# Hand detection & drawing handled in utils.hand_detector (uses its own MediaPipe instance)



def get_gesture(frame):
    hand_detected, updated_frame, bbox = detect_hand(frame)
    if not hand_detected or bbox is None:
        return 'None', 0.0, updated_frame

    x1, y1, w, h = bbox
    pad = ROI_PADDING // 2
    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)
    x2_pad = min(updated_frame.shape[1], x1 + w + 2 * pad)
    y2_pad = min(updated_frame.shape[0], y1 + h + 2 * pad)

    roi = updated_frame[y1_pad:y2_pad, x1_pad:x2_pad]
    if roi.size == 0:
        return 'None', 0.0, updated_frame

    # Draw ROI box
    cv2.rectangle(updated_frame, (x1_pad, y1_pad, x2_pad - x1_pad, y2_pad - y1_pad), GREEN, 2)

    img = preprocess(roi)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch, verbose=0)[0]
    confidence = float(np.max(pred) * 100)
    detected = CLASSES[np.argmax(pred)]
    return detected, confidence, updated_frame



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print('❌ Camera failed')
    exit()

# State
current_gesture = random.choice(CLASSES)
challenge_start = time.time()
hold_start = None
success_count = 0
state = 'waiting'

print(f"🎯 Show: {current_gesture} ({TIMEOUT_SEC}s timeout, hold {HOLD_REQUIRED}s)")
print('ESC/Q quit')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    detected, confidence, frame = get_gesture(frame)
    
    # Logic
    elapsed = time.time() - challenge_start
    time_left = max(0, TIMEOUT_SEC - elapsed)
    hold_elapsed = None
    status = None
    
    if confidence > 80 and detected == current_gesture:
        if state == 'waiting':
            state = 'holding'
            hold_start = time.time()
        elif state == 'holding':
            hold_elapsed = time.time() - hold_start
            if hold_elapsed >= HOLD_REQUIRED:
                success_count += 1
                status = 'Status: SUCCESS ✅'
                print(f"✅ #{success_count}/{NUM_SUCCESS}: {current_gesture}")
                if success_count >= NUM_SUCCESS:
                    print('🎉 CAPTCHA PASSED!')
                else:
                    current_gesture = random.choice(CLASSES)
                    challenge_start = time.time()
                    state = 'waiting'
                    print(f'🎯 Next: {current_gesture}')
    else:
        state = 'waiting'
    
    if time_left <= 0:
        print(f'⏰ Timeout! Next: {current_gesture}')
        current_gesture = random.choice(CLASSES)
        challenge_start = time.time()
        state = 'waiting'
    
    instruction = f'Show {current_gesture}'
    draw_exact_ui(frame, instruction, detected, confidence, status=status, hold_elapsed=hold_elapsed, time_left=time_left)
    
    cv2.imshow('Gesture CAPTCHA', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Complete.')



