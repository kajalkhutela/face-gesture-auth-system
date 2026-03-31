import cv2
import os

name = input("Enter person name: ")
path = f"face_module/dataset/raw/{name}"

os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to capture | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Images: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"{path}/{count}.jpg", frame)
        print("Saved:", count)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()