import cv2
import os
import numpy as np

input_dir = "face_module/dataset/processed"
output_dir = "face_module/dataset/augmented"

os.makedirs(output_dir, exist_ok=True)

def adjust_brightness(img, factor):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def add_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

for person in os.listdir(input_dir):
    in_path = os.path.join(input_dir, person)
    out_path = os.path.join(output_dir, person)

    os.makedirs(out_path, exist_ok=True)

    for img_name in os.listdir(in_path):
        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        base_name = img_name.split('.')[0]

        # Original
        cv2.imwrite(f"{out_path}/{base_name}_orig.jpg", img)

        # Flip
        cv2.imwrite(f"{out_path}/{base_name}_flip.jpg", cv2.flip(img, 1))

        # Brightness variations
        cv2.imwrite(f"{out_path}/{base_name}_bright.jpg", adjust_brightness(img, 1.3))
        cv2.imwrite(f"{out_path}/{base_name}_dark.jpg", adjust_brightness(img, 0.7))

        # Small rotations (VERY IMPORTANT)
        cv2.imwrite(f"{out_path}/{base_name}_rot1.jpg", rotate(img, 10))
        cv2.imwrite(f"{out_path}/{base_name}_rot2.jpg", rotate(img, -10))

        # Noise (simulate camera)
        cv2.imwrite(f"{out_path}/{base_name}_noise.jpg", add_noise(img))

print("Augmentation completed!")