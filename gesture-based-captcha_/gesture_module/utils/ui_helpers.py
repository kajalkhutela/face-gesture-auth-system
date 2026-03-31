import cv2
from config import *




def draw_exact_ui(frame, instruction, detected, confidence, state_status=None, hold_elapsed=None, time_left=None, status=None):
    """Minimal exact UI per spec."""
    y_pos = 30
    cv2.putText(frame, f"Instruction: {instruction}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    y_pos += 50
    det_color = GREEN if confidence > 80 else WHITE  # Simple green if high conf
    cv2.putText(frame, f"Detected: {detected}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, det_color, 2)
    y_pos += 50
    cv2.putText(frame, f"Confidence: {confidence:.0f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    
    if hold_elapsed is not None:
        y_pos += 40
        cv2.putText(frame, f"Hold: {hold_elapsed:.1f}s", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)
    if time_left is not None:
        y_pos += 40
        cv2.putText(frame, f"Time Left: {time_left:.0f}s", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)
    
    if status == "Status: SUCCESS ✅":
        y_pos += 50
        cv2.putText(frame, status, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)



