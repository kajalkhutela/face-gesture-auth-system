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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
original_face_enc = None
cooldown_end = 0
fps_deque = deque(maxlen=30)
frame_count = 0
frame_skip = 0
original_face_name = None
FACE_RECHECK_INTERVAL = 30

print("🎯 Face + Gesture 2FA v5.0 Anti-Spoof LIVE | Q=Quit")

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

def reverify_face(frame):
    """Anti-spoofing: verify same face is still present during gesture phase"""
    if original_face_name is None or original_face_enc is None:
        return False
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_frame)
        face_encs = face_recognition.face_encodings(rgb_frame, face_locs)
        
        if len(face_locs) == 1:
            current_enc = face_encs[0]
            # Distance to original saved encoding
            dist_orig = face_recognition.face_distance([original_face_enc], current_enc)[0]
            # Current name from known database
            distances = face_recognition.face_distance(known_encodings, current_enc)
            best_idx = np.argmin(distances)
            current_name = str(known_names[best_idx])
            
            is_same_person = (current_name == original_face_name and dist_orig < 0.5)
            if is_same_person:
                logging.info(f"🔒 Face recheck ✓ {current_name} (dist={dist_orig:.3f})")
            else:
                logging.warning(f"🚨 SPOOF FAIL: {current_name} != {original_face_name} (dist={dist_orig:.3f})")
            return is_same_person
        return False
    except Exception as e:
        logging.error(f"Face recheck ERROR: {e}")
        return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break
        
    frame = cv2.flip(frame, 1)
    
    frame_start = time.time()
    frame_count += 1
    frame_skip = (frame_skip + 1) % 3
    
    # 🔒 Anti-spoofing: Continuous face verification during gesture
    if state == STATE_GESTURE and (frame_count % FACE_RECHECK_INTERVAL == 0 or gesture_state == 'holding'):
        if not reverify_face(frame):
            status_text = "🚨 SPOOF DETECTED - ACCESS DENIED"
            logging.error(f"SPOOFING ATTEMPT BLOCKED during gesture {gesture_idx}")
            state = STATE_COOLDOWN
            cooldown_end = time.time() + 10
            continue
    
    avg_fps = np.mean(fps_deque) if fps_deque else 30.0
    try:
        fps = 1.0 / (time.time() - frame_start)
        fps_deque.append(fps)
    except:
        pass
    
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
                    print(f"Generated gesture sequence: {gesture_sequence}")
                    gesture_idx = 0
                    success_count = 0
                    retries = 0
                    gesture_state = 'waiting'
                    challenge_start = time.time()
                    original_face_name = face_name
                    original_face_enc = enc
                    state = STATE_GESTURE
                    status_text = f"Face ✓! Show first gesture: {gesture_sequence[0]}"
                    logging.info(f"Face auth: {face_name} conf={face_conf:.1%}, sequence={gesture_sequence}")
                else:
                    status_text = "Unknown face"
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
        print(f"Target: {target_gesture}, Detected: {detected} ({conf_pct:.1f}%)")
        
        # Gesture logic - no continuous face check
        elapsed = time.time() - challenge_start
        time_left = max(0, TIMEOUT_SEC - elapsed)
        
        if conf_pct > 60 and detected == target_gesture:
            print(f"✓ MATCH {detected} == {target_gesture} ({conf_pct:.1f}%)")
            if gesture_state == 'waiting':
                gesture_state = 'holding'
                hold_start = time.time()
            elif gesture_state == 'holding':
                hold_elapsed = time.time() - hold_start
                print(f"Holding for {hold_elapsed:.1f}s (need {HOLD_REQUIRED}s)")
                if hold_elapsed >= HOLD_REQUIRED:
                    success_count += 1
                    print(f"Gesture {gesture_idx+1} complete!")
                    gesture_state = 'waiting'
                    gesture_idx += 1
                    if gesture_idx >= NUM_CORRECT_REQUIRED:
                        status_text = f"🎉 {face_name} SUCCESS!"
                        state = STATE_SUCCESS
                        cooldown_end = time.time() + 5
                    else:
                        challenge_start = time.time()
        else:
            gesture_state = 'waiting'
            if detected != 'None' and detected != 'Error':
                print(f"❌ No match: {detected} != {target_gesture}")
        
        if time_left <= 0:
            print(f"⏰ Timeout")
            gesture_state = 'waiting'
            gesture_idx += 1
            if gesture_idx >= NUM_CORRECT_REQUIRED:
                retries += 1
                if retries >= 3:
                    status_text = "❌ ACCESS DENIED"
                    state = STATE_COOLDOWN
                    cooldown_end = time.time() + 10
                else:
                    gesture_sequence = random.sample(CLASSES, NUM_CORRECT_REQUIRED)
                    gesture_idx = 0
                    success_count = 0
                    challenge_start = time.time()
                    status_text = f"Retry {retries+1}/3"
            else:
                challenge_start = time.time()
        
        instruction = f"Gesture {gesture_idx+1}/{NUM_CORRECT_REQUIRED}: {target_gesture}"
        hold_elapsed = (time.time() - hold_start) if gesture_state == 'holding' else 0.0
        draw_exact_ui(frame, instruction, detected, conf_pct, time_left, hold_elapsed)
    
    elif state == STATE_SUCCESS:
        status_text = "SUCCESS! Cooldown..."
        if time.time() > cooldown_end:
            state = STATE_FACE
            face_name = None
    
    elif state == STATE_COOLDOWN:
        status_text = "DENIED - Resetting..."
        if time.time() > cooldown_end:
            state = STATE_FACE
    
    # Single GLOBAL UI - FIXED duplication
    cv2.putText(frame, status_text or "Face detection...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    if original_face_name:
        live_status = " LIVE ✓" if state == STATE_GESTURE else ""
        cv2.putText(frame, f"Auth: {original_face_name}{live_status}", (10, frame.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    elif face_name:
        color = (0,255,0) if face_conf > 0.5 else (0,0,255)
        cv2.putText(frame, f"Face: {face_name} ({face_conf:.0f}%)", (10, frame.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, f"Gestures: {success_count}/{NUM_CORRECT_REQUIRED}", (10, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, f"Retries: {retries}/3 | FPS: {avg_fps:.1f} | Q=Quit", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow('Face + Gesture 2FA - FIXED', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Original version restored & duplication fixed")
