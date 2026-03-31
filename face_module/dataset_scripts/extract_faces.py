import cv2
import os
from mtcnn import MTCNN

detector = MTCNN()

input_dir = "dataset/raw"
output_dir = "dataset/processed"

os.makedirs(output_dir, exist_ok=True)

for person in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person)
    save_path = os.path.join(output_dir, person)

    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        results = detector.detect_faces(img)

        if len(results) == 0:
            continue

        x, y, w, h = results[0]['box']

        # Fix negative coordinates
        x, y = max(0, x), max(0, y)

        face = img[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (160, 160))
            cv2.imwrite(os.path.join(save_path, img_name), face)
        except:
            continue

print("Face extraction completed!")