import cv2
import numpy as np
import face_recognition
import os

# -------------------------------
# 🚀 PATHS
# -------------------------------
dataset_path = "face_module/dataset/augmented"  # Use augmented images for robustness
embedding_path = "face_module/dataset/embeddings"
os.makedirs(embedding_path, exist_ok=True)

# -------------------------------
# 🔥 LOAD EMBEDDINGS
# -------------------------------
encodings_file = os.path.join(embedding_path, "encodings.npy")
names_file = os.path.join(embedding_path, "names.npy")

if os.path.exists(encodings_file) and os.path.exists(names_file):
    known_encodings = np.load(encodings_file)
    known_names = np.load(names_file)
    print(f"✅ Loaded {len(known_names)} known identities")
    if len(known_encodings) == 0:
        print("⚠️ No encodings found!")
        import sys
        sys.exit(1)
else:
    print("⚠️ No embeddings found. Run the embedding script first!")
    import sys
    sys.exit(1)

# -------------------------------
# 🔥 STABILITY VARIABLES
# -------------------------------
last_name = "Unknown"
stable_count = 0
final_name = "Detecting..."
threshold = 0.4  # lower = stricter, adjust for your setup

# -------------------------------
# 🔹 CAMERA SETUP
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    import sys
    sys.exit(1)

print("📸 Camera started...")

# -------------------------------
# 🔹 MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        continue

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known encodings
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        score = distances[best_match_index]

        name = "Unknown"
        if score < threshold:
            name = known_names[best_match_index]

        # -------------------------------
        # 🔹 STABILITY LOGIC
        # -------------------------------
        if name == last_name:
            stable_count += 1
        else:
            stable_count = 0
            last_name = name

        if stable_count > 3:
            final_name = name
        else:
            final_name = "Detecting..."

        # -------------------------------
        # 🔹 DISPLAY
        # -------------------------------
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label
        label = f"{final_name} ({score:.2f})"
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 🔹 CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("📴 Camera released. Program ended.")