import cv2
import os
import time
from utils.hand_detector import detect_hand

print("🚀 FAST GESTURE DATA COLLECTION")
print("• SPACEBAR = Instant Save")
print("• Hold gesture 1-2s → SPACE → Repeat")
print("• ESC = Stop & Save")
print()

gesture_name = input("Enter Gesture (Fist/One/Two/Palm/ThumbsUp/OK): ").strip()

save_path = f"dataset/{gesture_name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
img_count = len([f for f in os.listdir(save_path) if f.endswith('.jpg')])

last_save_time = time.time()

print(f"📁 Saving to: {save_path}")
print(f"Current count: {img_count}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI + Hand Detection
    hand_detected, frame = detect_hand(frame)
    x1, y1, x2, y2 = 200, 100, 450, 350
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    current_time = time.time()
    
    # Auto-save DISABLED - manual SPACEBAR only
    pass

    # MANUAL FAST SAVE (SPACEBAR)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar - INSTANT SAVE
        if roi.size > 0:
            img_name = f"{save_path}/{gesture_name}_{img_count:03d}.jpg"
            cv2.imwrite(img_name, roi)
            print(f"✅ FAST SAVE #{img_count:03d}: {gesture_name}")
            img_count += 1
            last_save_time = current_time
        else:
            print("❌ Empty ROI - move hand into box")
    
    elif key == 27:  # ESC
        break

    # Display info
    cv2.putText(frame, f"{gesture_name} - {img_count} imgs", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    status = "🟢 HAND OK" if hand_detected else "🔴 NO HAND"
    cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "SPACE=Save | ESC=Stop", (10, frame.shape[0]-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("FAST Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
print(f"🎉 Saved {img_count} images for {gesture_name}")
