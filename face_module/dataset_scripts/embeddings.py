import os
import numpy as np
from deepface import DeepFace

print("🚀 Script started...")

dataset_path = "face_module/dataset/augmented"
save_path = "face_module/dataset/embeddings"

os.makedirs(save_path, exist_ok=True)

all_embeddings = []
all_names = []

# -------------------------------
# LOOP THROUGH PERSONS
# -------------------------------
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    print(f"\n👤 Processing: {person}")

    person_embeddings = []
    image_list = os.listdir(person_path)

    # 🔥 LIMIT images (avoid overload)
    image_list = image_list[:120]

    for i, img_name in enumerate(image_list):
        img_path = os.path.join(person_path, img_name)

        print(f"➡️ {person} | {i+1}/{len(image_list)}")

        try:
            result = DeepFace.represent(
                img_path,
                model_name="Facenet",   # 🔥 CHANGED HERE
                enforce_detection=False
            )

            embedding = result[0]["embedding"]
            person_embeddings.append(embedding)

        except Exception as e:
            print("❌ Error:", e)
            continue

    # -------------------------------
    # CHECK EMPTY
    # -------------------------------
    if len(person_embeddings) == 0:
        print("⚠️ No embeddings for:", person)
        continue

    # -------------------------------
    # AVERAGE EMBEDDING (VERY IMPORTANT)
    # -------------------------------
    avg_embedding = np.mean(person_embeddings, axis=0)

    all_embeddings.append(avg_embedding)
    all_names.append(person)

# -------------------------------
# SAVE
# -------------------------------
np.save(os.path.join(save_path, "encodings.npy"), all_embeddings)
np.save(os.path.join(save_path, "names.npy"), all_names)

print("\n✅ Embeddings saved successfully!")
print("📁 Saved at:", save_path)
print("👥 Total persons:", len(all_names))