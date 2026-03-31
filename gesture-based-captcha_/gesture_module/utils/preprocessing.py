import cv2
import numpy as np

def preprocess(img):
    """Preprocess ROI for model input."""
    # Keep COLOR for better accuracy (model expects RGB, 96x96 as trained)
    img = cv2.resize(img, (96, 96))
    img = img.astype(np.float32) / 255.0
    return img



