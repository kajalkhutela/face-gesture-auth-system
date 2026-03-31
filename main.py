
"""
Production Face + Gesture 2FA - Bug-free v3.0
Uses main_gesture.py logic + robust face verification
"""

import cv2
import numpy as np
import time
import random
import tensorflow as tf
import face_recognition
import sys
import os
import logging
from datetime import datetime
from collections import deque

# Gesture module imports
sys.path.append('gesture-based-captcha_/gesture_module')
from utils.hand_detector import detect_hand
from utils.preprocessing import preprocess
from utils.ui_helpers import draw_exact_ui
from config import MODEL_PATH, CLASSES, NUM_CORRECT_REQUIRED, TIMEOUT_SEC, HOLD_REQUIRED, ROI_PADDING

# Face paths
FACE_ENCODINGS_PATH = "face_module/dataset/embeddings/encodings.npy"
FACE_NAMES_PATH = "face_module/dataset/embeddings/names.npy"

# Logging
logging.basicConfig(filename='auth.log', level=logging.INFO, 
                   format='%(asctime)s,%(msecs)03d %(message)s', datefmt='%H:%M:%S')

# Load models & data
gesture_model = tf.keras.models.load_model(MODEL_PATH)
known_encodings = np.load(FACE_ENCODINGS_PATH)
known_names = np.load(FACE_NAMES_PATH, allow_pickle=True)

print(f"✅ Loaded {len(known_names)} faces, {len(CLASSES)} gestures")

# Camera setup with error check
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera failed to open")
    exit(1)

# Robust state machine
STATE_FACE = 0
STATE_GESTURE = 1
STATE_SUCCESS = 2
STATE_COOLDOWN = 3

state = STATE_FACE
gesture_sequence = []
gesture_idx = 0
retries = 0
success_count = 0
challenge_start = 0
hold_start = 0
gesture_state = 'waiting'
face_name = None
face_conf = 0.0
cooldown_end = 0
fps_deque = deque(maxlen=30)
frame_count = 0  # FPS safety

print("🎯 Face + Gesture 2FA | Q=Quit")

def get_gesture_safe(frame):
    """Safe wrapper for gesture detection"""
    try:
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

        cv2.rectangle(updated_frame, (x1_pad, y1_pad, x2_pad-x1_pad, y2_pad-y1_pad), (0,255,0), 2)

        img = preprocess(roi)
        img_batch = np.expand_dims(img, axis=0)
        pred = gesture_model.predict(img_batch, verbose=0)[0]
        confidence = np.max(pred) * 100
        detected = CLASSES[np.argmax(pred)]
        return detected, confidence, updated_frame
    except Exception as e:
        print(f"Gesture error: {e}")
        return 'Error', 0.0, frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break
        
    frame = cv2.flip(frame, 1)
    frame_start = time.time()
    frame_count += 1
    
    status_text = ""
    
    # ===== FACE VERIFICATION =====
    if state == STATE_FACE:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_frame)
            face_encs = face_recognition.face_encodings(rgb_frame, face_locs)
            
            if len(face_locs) == 1:
                enc = face_encs[0]
                distances = face_recognition.face_distance(known_encodings, enc)
                best_idx = np.argmin(distances)
                distance = distances[best_idx]
                
                if distance < 0.45:
                    face_name = str(known_names[best_idx])
                    face_conf = max(0.0, 1 - distance)
                    gesture_sequence = random.sample(CLASSES, NUM_CORRECT_REQUIRED)
                    gesture_idx = 0
                    success_count = 0
                    retries = 0
                    gesture_state = 'waiting'
                    challenge_start = time.time()
                    state = STATE_GESTURE
                    status_text = f"Face ✓: {face_name}"
                    logging.info(f"Face auth: {face_name} conf={face_conf:.1%}")
                else:
                    status_text = "Unknown face (dist>0.6)"
            elif len(face_locs) > 1:
                status_text = f"Too many faces ({len(face_locs)})"
            else:
                status_text = "No face detected"
        except Exception as e:
            status_text = f"Face error: {str(e)[:30]}"
    
    # ===== GESTURE CHALLENGE =====
    elif state == STATE_GESTURE:
        target_gesture = gesture_sequence[gesture_idx]
        
        detected, conf_pct, frame = get_gesture_safe(frame)
        
        elapsed = time.time() - challenge_start
        time_left = max(0, TIMEOUT_SEC - elapsed)
        hold_elapsed = None
        ui_status = None
        
        if conf_pct > 80 and detected == target_gesture:
            if gesture_state == 'waiting':
                gesture_state = 'holding'
                hold_start = time.time()
            elif gesture_state == 'holding':
                hold_elapsed = time.time() - hold_start
                if hold_elapsed >= HOLD_REQUIRED:
                    success_count += 1
                    ui_status = "SUCCESS ✅"
                    logging.info(f"Gesture OK #{success_count}: {target_gesture} (hold={hold_elapsed:.1f}s)")
                    
                    gesture_idx += 1
                    if gesture_idx >= NUM_CORRECT_REQUIRED:
                        status_text = f"🎉 {face_name} SUCCESS!"
                        logging.info(f"Full auth success: {face_name}")
                        state = STATE_SUCCESS
                        cooldown_end = time.time() + 5  # 5s success cooldown
                    else:
                        # Next gesture
                        challenge_start = time.time()
                        gesture_state = 'waiting'
        else:
            gesture_state = 'waiting'
        
        # Timeout handling
        if time_left <= 0 and gesture_state != 'holding':
            print(f"⏰ Timeout on {target_gesture}")
            gesture_idx += 1
            if gesture_idx >= NUM_CORRECT_REQUIRED:
                # Gesture challenge failed
                retries += 1
                if retries >= 3:
                    status_text = "❌ ACCESS DENIED (3 retries)"
                    logging.warning(f"Auth failed: {face_name} after {retries} retries")
                    state = STATE_COOLDOWN
                    cooldown_end = time.time() + 10
                else:
                    # Retry challenge
                    gesture_sequence = random.sample(CLASSES, NUM_CORRECT_REQUIRED)
                    gesture_idx = 0
                    success_count = 0
                    challenge_start = time.time()
                    gesture_state = 'waiting'
                    status_text = f"Retry {retries+1}/3"
            else:
                challenge_start = time.time()
                gesture_state = 'waiting'
        
        instruction = f"Gesture {gesture_idx+1}/{NUM_CORRECT_REQUIRED}: {target_gesture}"
        draw_exact_ui(frame, instruction, detected, conf_pct, status=ui_status,
                     hold_elapsed=hold_elapsed, time_left=time_left)
    
    # ===== SUCCESS/COOLDOWN =====
    elif state == STATE_SUCCESS:
        remain = max(0, cooldown_end - time.time())
        status_text = f"SUCCESS! Cooldown: {remain:.0f}s"
        if remain <= 0:
            state = STATE_FACE
            face_name = None
    
    elif state == STATE_COOLDOWN:
        remain = max(0, cooldown_end - time.time())
        status_text = f"DENIED - Reset in: {remain:.0f}s"
        if remain <= 0:
            state = STATE_FACE
    
    # ===== GLOBAL UI & FPS (safe calculation) =====
    try:
        elapsed_frame = time.time() - frame_start
        if elapsed_frame > 0:
            fps = 1 / elapsed_frame
            fps_deque.append(fps)
    except:
        fps = 0.0
    
    avg_fps = np.mean(fps_deque) if fps_deque else 0
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    if face_name:
        cv2.putText(frame, f"Face: {face_name} [{face_conf:.0%}]", (10, frame.shape[0]-100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    cv2.putText(frame, f"Progress: {success_count}/{NUM_CORRECT_REQUIRED}", (10, frame.shape[0]-70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Retries: {retries}/3", (10, frame.shape[0]-40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if retries else (255,255,255), 2)
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    cv2.imshow('Face + Gesture 2FA v3.0', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Q or ESC
        break

cap.release()
cv2.destroyAllWindows()
logging.info("Session ended")
print("✅ Clean shutdown")
