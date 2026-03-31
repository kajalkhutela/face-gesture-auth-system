import cv2
import os

input_dir = "face_module/dataset/processed"
min_blur_threshold = 30 # lower = more blur

for person in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur detection
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_value < min_blur_threshold:
            os.remove(img_path)
            print("Removed blurry:", img_name)