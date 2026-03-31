import face_recognition
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "../../face_module/dataset/processed")
save_path = os.path.join(BASE_DIR, "../../face_module/dataset/embeddings")

os.makedirs(save_path, exist_ok=True)

encodings = []
names = []

print("🚀 Encoding faces...")

total_images = 0
valid_images = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    print(f"\n👤 Processing: {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        total_images += 1

        try:
            image = face_recognition.load_image_file(img_path)

            # Detect faces
            face_locations = face_recognition.face_locations(image)

            # ❌ Skip if no face or multiple faces
            if len(face_locations) != 1:
                continue

            face_enc = face_recognition.face_encodings(image, face_locations)

            if len(face_enc) > 0:
                encodings.append(face_enc[0])
                names.append(person)
                valid_images += 1

        except Exception as e:
            print(f"❌ Error in {img_name}: {e}")
            continue

print("\n📊 Summary:")
print(f"Total images: {total_images}")
print(f"Valid encodings: {valid_images}")

print("\n💾 Saving encodings...")

np.save(os.path.join(save_path, "encodings.npy"), encodings)
np.save(os.path.join(save_path, "names.npy"), names)

print("✅ Done! Embeddings saved successfully.")