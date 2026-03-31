import cv2
import numpy as np
import face_recognition
import os
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
DATASET_PATH = "face_module/dataset/augmented"
ENCODINGS_PATH = "face_module/dataset/encodings.npy"
NAMES_PATH = "face_module/dataset/names.npy"
THRESHOLD = 0.4  # similarity threshold

# -------------------------------
# LOAD OR CREATE EMBEDDINGS
# -------------------------------
if os.path.exists(ENCODINGS_PATH) and os.path.exists(NAMES_PATH):
    known_encodings = np.load(ENCODINGS_PATH, allow_pickle=True)
    known_names = np.load(NAMES_PATH, allow_pickle=True)
    print("✅ Loaded saved embeddings.")
else:
    print("⚡ Generating embeddings from dataset...")
    known_encodings = []
    known_names = []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        embeddings_list = []
        images = os.listdir(person_path)[:100]  # limit images
        for img_name in images:
            img_path = os.path.join(person_path, img_name)
            try:
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    embeddings_list.append(encs[0])
            except Exception as e:
                print("❌ Error:", e)
        if embeddings_list:
            avg_encoding = np.mean(embeddings_list, axis=0)
            known_encodings.append(avg_encoding)
            known_names.append(person)

    np.save(ENCODINGS_PATH, known_encodings)
    np.save(NAMES_PATH, known_names)
    print("✅ Embeddings created and saved.")

# -------------------------------
# SIMPLE LIVENESS DETECTION
# -------------------------------
def is_live_face(face_frame, blink_history=[0,0,0]):
    # Very basic: checks if eyes are open at least once in last 3 frames
    face_gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eyes_cascade.detectMultiScale(face_gray, 1.1, 4)
    blink_history.pop(0)
    blink_history.append(1 if len(eyes) > 0 else 0)
    if sum(blink_history) > 0:
        return True
    return False

# -------------------------------
# CAMERA SETUP
# -------------------------------
cap = cv2.VideoCapture(0)
blink_history = [0,0,0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_dist_index = np.argmin(distances)
        name = "Unknown"
        if distances[min_dist_index] < THRESHOLD:
            # Check liveness
            face_img = frame[top:bottom, left:right]
            if is_live_face(face_img, blink_history):
                name = known_names[min_dist_index]

        # Draw green box and name
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition + Liveness", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()