# Gesture CAPTCHA Config

# Paths
MODEL_PATH = "gesture-based-captcha_/models/gesture_model.h5"
DATA_DIR = "../dataset"  # relative to gesture_module

# Model
CLASSES = ["Fist", "One", "Two", "Palm", "ThumbsUp"]
IMG_SHAPE = (96, 96, 3)  # HWC from training
CONF_THRESHOLD = 0.85
PREDICT_BATCH_SIZE = 1

# CAPTCHA
NUM_CORRECT_REQUIRED = 3
MAX_CHALLENGES_PER_SESSION = 5
TIMEOUT_SEC = 10  # per challenge

# UI/ROI
ROI_PADDING = 50  # pixels around hand bbox
FONT_SCALE = 0.7
BIGGER_FONT = 1.0
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# CAPTCHA Hold
HOLD_REQUIRED = 3.0

# Camera
CAM_INDEX = 0
FLIP_H = True

