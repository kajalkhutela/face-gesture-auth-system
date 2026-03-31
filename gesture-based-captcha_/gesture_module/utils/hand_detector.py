import cv2
import mediapipe as mp
import numpy as np  # for bbox calc

# Global MediaPipe instance for efficiency
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=2,  # Allow 2 hands
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

def detect_hand(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    bbox = None
    if results.multi_hand_landmarks:
        # Pick the first/largest hand for simplicity
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Compute bbox from landmarks
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        bbox = (x1, y1, x2 - x1, y2 - y1)  # x,y,w,h
    
    hand_detected = bbox is not None
    return hand_detected, frame, bbox



