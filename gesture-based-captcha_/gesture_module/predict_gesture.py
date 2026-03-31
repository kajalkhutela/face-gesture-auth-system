import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess
from utils.hand_detector import detect_hand

model = load_model("models/gesture_model.h5")

classes = ["Fist", "One", "Two", "Palm", "ThumbsUp"]

cap = cv2.VideoCapture(0)

print("Q=Quit | Hand in green box")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    hand_detected, frame = detect_hand(frame)

    x1, y1, x2, y2 = 200, 100, 450, 350
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    if hand_detected and roi.size > 0:
        img = preprocess(roi)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)
        gesture = classes[np.argmax(pred)]
        confidence = np.max(pred) * 100
        cv2.putText(frame, f"{gesture}: {confidence:.0f}%", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No hand", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Prediction - FIXED", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
