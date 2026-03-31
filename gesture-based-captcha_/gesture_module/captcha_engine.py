import random
import time
from config import CLASSES as gestures, TIMEOUT_SEC, NUM_CORRECT_REQUIRED, HOLD_REQUIRED

class GestureCaptcha:
    def __init__(self):
        self.gestures = gestures
        self.current_target = None
        self.state = None  # 'waiting_detection', 'holding'
        self.challenge_start = None
        self.hold_start = None
        self.is_active = False
        self.session_successes = 0
        self.max_attempts = NUM_CORRECT_REQUIRED
        self.time_limit = 15

    def generate_challenge(self):
        self.current_target = random.choice(self.gestures)
        self.state = 'waiting_detection'
        self.challenge_start = time.time()
        self.hold_start = None
        self.is_active = True
        return f"New challenge generated: {self.current_target}"

    def verify_gesture(self, predicted):
        if not self.is_active:
            return 'inactive'

        # Timeout check
if time.time() - self.challenge_start > self.time_limit:
            self.reset_challenge()
            return 'timeout'

        if self.state == 'waiting_detection':
            if predicted == self.current_target:
                self.state = 'holding'
                self.hold_start = time.time()
                return 'hold_started'
            else:
                return 'wrong_gesture'
        elif self.state == 'holding':
            hold_elapsed = time.time() - self.hold_start
            if hold_elapsed >= HOLD_REQUIRED:
                self.session_successes += 1
                print(f"✅ SUCCESS! Total: {self.session_successes}/{self.max_attempts}")
                if self.session_successes >= self.max_attempts:
                    self.is_active = False
                    return 'captcha_complete'
                self.generate_challenge()
                return 'success_next'
            else:
                return 'holding'
        return 'invalid'

    def reset_challenge(self):
        self.is_active = False
        self.current_target = None
        self.state = None
        self.challenge_start = None
        self.hold_start = None

    def time_left(self):
        if not self.challenge_start:
            return self.time_limit
        elapsed = time.time() - self.challenge_start
        return max(0.0, self.time_limit - elapsed)

    def is_time_over(self):
        return bool(self.challenge_start and (time.time() - self.challenge_start > self.time_limit))

    def get_instruction(self):
        return self.current_target or "?"

    def verify(self, gesture, stability):
        if stability < 0.7:
            return False
        result = self.verify_gesture(gesture)
        return result in ['success_next', 'captcha_complete']

    def reset(self):
        self.reset_challenge()

    def get_confidence(self):
        return 0.0  # Use external stability in main
