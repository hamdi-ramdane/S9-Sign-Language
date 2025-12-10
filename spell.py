"""
ASL Alphabet Spelling Mode - Two-Hand Control
Right hand: Letter prediction
Left hand: Pinch gestures to control
  - Index + Thumb pinch = Add letter
  - Middle + Thumb pinch = Backspace (delete last)
  - Pinky + Thumb pinch = Reset (clear all)

CONTROLS:
=========
   SPACE  - Start/Pause recognition
   C      - Clear typed text
   V      - Toggle visualization (None/Hands/Full)
   BACKSPACE - Delete last character
   Q/ESC  - Quit

"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import os
import time
import json
from collections import deque

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "asl_alphabet_landmarks.keras"
CONFIG_PATH = MODEL_DIR / "model_config.json"
SCALER_PATH = MODEL_DIR / "scaler_params.json"



# Camera settings
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAMERA_FPS = 30

# Display settings - Full HD
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
UI_WIDTH = 550  # Right panel - wider for better spacing

# Recognition settings
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5
HISTORY_SIZE = 100

# Pinch detection settings
PINCH_THRESHOLD = 0.06  # Distance threshold for pinch (normalized)
PINCH_FRAMES_REQUIRED = 3  # Frames to confirm pinch

# ============================================================================
# LOAD MODEL & CONFIG
# ============================================================================
MODEL_LOADED = False
model = None
CLASS_NAMES = []
SCALER_MEAN = None
SCALER_SCALE = None

# Load config
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        MODEL_CONFIG = json.load(f)
    CLASS_NAMES = MODEL_CONFIG.get("class_names", [])
    print(f"✅ Config loaded: {len(CLASS_NAMES)} classes")
    print(f"   Classes: {CLASS_NAMES}")
else:
    print(f"❌ Config not found: {CONFIG_PATH}")
    CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'I', 'K', 'L', 'O', 'U', 'del', 'space']

# Load scaler
if SCALER_PATH.exists():
    with open(SCALER_PATH) as f:
        scaler_params = json.load(f)
    SCALER_MEAN = np.array(scaler_params['mean'])
    SCALER_SCALE = np.array(scaler_params['scale'])
    print(f"✅ Scaler loaded")
else:
    print(f"❌ Scaler not found")

# Load model
try:
    import tensorflow as tf
    
    if MODEL_PATH.exists():
        print(f"📦 Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded")
        MODEL_LOADED = True
    else:
        print(f"❌ Model not found: {MODEL_PATH}")
        alt_path = OUTPUT_DIR / "asl_alphabet_landmarks.keras"
        if alt_path.exists():
            model = tf.keras.models.load_model(alt_path)
            MODEL_LOADED = True
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_hand_landmarks(hand_landmarks):
    """Extract normalized landmarks from MediaPipe hand."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    
    landmarks = np.array(landmarks)
    
    # Center around wrist
    wrist = landmarks[0]
    landmarks_centered = landmarks - wrist
    
    # Scale normalize
    max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
    if max_dist > 0:
        landmarks_normalized = landmarks_centered / max_dist
    else:
        landmarks_normalized = landmarks_centered
    
    return landmarks_normalized.flatten().astype(np.float32)


def apply_scaler(features):
    """Apply StandardScaler transformation."""
    if SCALER_MEAN is not None and SCALER_SCALE is not None:
        return (features - SCALER_MEAN) / SCALER_SCALE
    return features


def detect_pinches(hand_landmarks):
    """
    Detect pinch gestures between thumb and fingers.
    
    Returns:
        index_pinch: bool - True if index finger touching thumb (add letter)
        middle_pinch: bool - True if middle finger touching thumb (backspace)
        pinky_pinch: bool - True if pinky finger touching thumb (reset)
        index_dist, middle_dist, pinky_dist: float - Distances
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)
    
    # Landmark indices:
    # 4 = thumb tip
    # 8 = index finger tip
    # 12 = middle finger tip
    # 20 = pinky finger tip
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    pinky_tip = landmarks[20]
    
    # Calculate distances
    index_dist = np.linalg.norm(index_tip - thumb_tip)
    middle_dist = np.linalg.norm(middle_tip - thumb_tip)
    pinky_dist = np.linalg.norm(pinky_tip - thumb_tip)
    
    # Check if pinching
    index_pinch = index_dist < PINCH_THRESHOLD
    middle_pinch = middle_dist < PINCH_THRESHOLD
    pinky_pinch = pinky_dist < PINCH_THRESHOLD
    
    return index_pinch, middle_pinch, pinky_pinch, index_dist, middle_dist, pinky_dist


def draw_text_with_bg(frame, text, pos, scale=1.0, color=(255, 255, 255), 
                      bg_color=(0, 0, 0), thickness=2, padding=5):
    """Draw text with background."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - padding, y - th - padding), 
                  (x + tw + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_confidence_bar(frame, x, y, width, height, confidence, color=(0, 255, 0)):
    """Draw a confidence bar."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    fill_width = int(width * confidence)
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 2)


def draw_hand_landmarks_on_frame(frame, hand_landmarks, viz_mode):
    """Draw hand landmarks based on visualization mode."""
    if viz_mode == "none":
        return
    
    if viz_mode == "hands":
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(100, 255, 100), thickness=2)
        )
    elif viz_mode == "full":
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


def get_top_predictions(predictions, class_names, top_k=3):
    """Get top-k predictions with class names and confidences."""
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'confidence': predictions[idx],
            'index': idx
        })
    return results


class PredictionSmoother:
    """Smooth predictions using moving average."""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions_buffer = deque(maxlen=window_size)
    
    def add_prediction(self, prediction):
        self.predictions_buffer.append(prediction)
    
    def get_smoothed_prediction(self):
        if len(self.predictions_buffer) == 0:
            return None
        return np.mean(self.predictions_buffer, axis=0)
    
    def clear(self):
        self.predictions_buffer.clear()


class PinchTrigger:
    """Track pinch gestures for triggering actions."""
    
    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.index_count = 0
        self.middle_count = 0
        self.pinky_count = 0
        self.was_index_pinch = False
        self.was_middle_pinch = False
        self.was_pinky_pinch = False
    
    def update(self, index_pinch, middle_pinch, pinky_pinch):
        """
        Update with pinch detection results.
        Returns:
            add_trigger: bool - True when index pinch confirmed (add letter)
            backspace_trigger: bool - True when middle pinch confirmed (backspace)
            reset_trigger: bool - True when pinky pinch confirmed (reset)
        """
        add_trigger = False
        backspace_trigger = False
        reset_trigger = False
        
        # Index pinch (add letter)
        if index_pinch:
            self.index_count += 1
        else:
            if self.was_index_pinch:
                self.was_index_pinch = False
            self.index_count = 0
        
        if self.index_count >= self.required_frames and not self.was_index_pinch:
            self.was_index_pinch = True
            add_trigger = True
        
        # Middle pinch (backspace)
        if middle_pinch:
            self.middle_count += 1
        else:
            if self.was_middle_pinch:
                self.was_middle_pinch = False
            self.middle_count = 0
        
        if self.middle_count >= self.required_frames and not self.was_middle_pinch:
            self.was_middle_pinch = True
            backspace_trigger = True
        
        # Pinky pinch (reset)
        if pinky_pinch:
            self.pinky_count += 1
        else:
            if self.was_pinky_pinch:
                self.was_pinky_pinch = False
            self.pinky_count = 0
        
        if self.pinky_count >= self.required_frames and not self.was_pinky_pinch:
            self.was_pinky_pinch = True
            reset_trigger = True
        
        return add_trigger, backspace_trigger, reset_trigger
    
    def clear(self):
        self.index_count = 0
        self.middle_count = 0
        self.pinky_count = 0
        self.was_index_pinch = False
        self.was_middle_pinch = False
        self.was_pinky_pinch = False


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    if not MODEL_LOADED:
        print("❌ Cannot start - model not loaded!")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    # Initialize MediaPipe for 2 hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # State
    recognizing = False
    viz_mode = "hands"
    viz_modes = ["none", "hands", "full"]
    viz_idx = 1
    
    # Prediction & trigger
    smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW)
    pinch_trigger = PinchTrigger(required_frames=PINCH_FRAMES_REQUIRED)
    
    # Typed text
    typed_text = ""
    
    # FPS tracking
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Inference time
    inference_times = deque(maxlen=30)
    
    # Last added letter for visual feedback
    last_added = None
    last_added_time = 0
    
    print("\n" + "="*60)
    print("ASL SPELLING MODE - TWO-HAND CONTROL")
    print("="*60)
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print("-"*60)
    print("RIGHT HAND = Letter prediction")
    print("LEFT HAND  = Index+Thumb   -> ADD letter")
    print("             Middle+Thumb  -> BACKSPACE")
    print("             Pinky+Thumb   -> RESET all")
    print("-"*60)
    print("SPACE: Start/Pause | V: Viz | C: Clear | Q/ESC: Quit")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        # Create display
        display = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        
        # Camera feed
        cam_width = DISPLAY_WIDTH - UI_WIDTH
        cam_display = cv2.resize(frame, (cam_width, DISPLAY_HEIGHT))
        display[:, :cam_width] = cam_display
        
        # Track state
        right_hand = None
        left_hand = None
        current_pred = None
        add_trigger = False
        backspace_trigger = False
        reset_trigger = False
        index_pinch = False
        middle_pinch = False
        pinky_pinch = False
        index_dist = 1.0
        middle_dist = 1.0
        pinky_dist = 1.0
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                   results.multi_handedness):
                # Get hand label (note: flipped due to mirror)
                label = handedness.classification[0].label
                
                # Mirror flip: "Left" in MediaPipe = right hand in camera
                if label == "Left":
                    right_hand = hand_landmarks
                else:
                    left_hand = hand_landmarks
                
                # Draw landmarks
                draw_hand_landmarks_on_frame(display[:, :cam_width], 
                                            hand_landmarks, viz_mode)
        
        # Process right hand (letter prediction)
        if right_hand and recognizing and MODEL_LOADED:
            start_time = time.time()
            
            features = extract_hand_landmarks(right_hand)
            features_scaled = apply_scaler(features)
            
            prediction = model.predict(features_scaled.reshape(1, -1), verbose=0)[0]
            
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            smoother.add_prediction(prediction)
            smoothed = smoother.get_smoothed_prediction()
            
            if smoothed is not None:
                current_pred = get_top_predictions(smoothed, CLASS_NAMES, top_k=3)
        
        # Process left hand (pinch triggers)
        if left_hand and recognizing:
            index_pinch, middle_pinch, pinky_pinch, index_dist, middle_dist, pinky_dist = detect_pinches(left_hand)
            add_trigger, backspace_trigger, reset_trigger = pinch_trigger.update(index_pinch, middle_pinch, pinky_pinch)
            
            # Add letter on index pinch
            if add_trigger and current_pred:
                top_class = current_pred[0]['class']
                top_conf = current_pred[0]['confidence']
                
                if top_conf >= CONFIDENCE_THRESHOLD:
                    if top_class == 'space':
                        typed_text += ' '
                    elif top_class == 'del':
                        typed_text = typed_text[:-1] if typed_text else ""
                    else:
                        typed_text += top_class
                    
                    if len(typed_text) > HISTORY_SIZE:
                        typed_text = typed_text[-HISTORY_SIZE:]
                    
                    last_added = top_class
                    last_added_time = time.time()
                    print(f"✅ Added: '{top_class}' -> '{typed_text}'")
            
            # Backspace on middle pinch
            if backspace_trigger:
                if typed_text:
                    typed_text = typed_text[:-1]
                    print(f"⌫ Backspace -> '{typed_text}'")
            
            # Reset on pinky pinch
            if reset_trigger:
                typed_text = ""
                smoother.clear()
                print("🗑️  Reset text")
        
        if not left_hand:
            pinch_trigger.clear()
        
        if not right_hand and recognizing:
            smoother.clear()
        
        # ====================================================================
        # DRAW UI PANEL
        # ====================================================================
        ui_x = DISPLAY_WIDTH - UI_WIDTH
        cv2.rectangle(display, (ui_x, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (25, 25, 30), -1)
        
        # Margin for UI elements
        margin = 40
        
        # Title
        draw_text_with_bg(display, "SPELL MODE", (ui_x + margin, 70), 
                         scale=1.5, color=(0, 255, 255), thickness=3)
        draw_text_with_bg(display, "Pinch Gestures Control", (ui_x + margin, 115), 
                         scale=0.8, color=(0, 200, 200), thickness=2)
        
        # Divider line
        cv2.line(display, (ui_x + margin, 145), (DISPLAY_WIDTH - margin, 145), (60, 60, 70), 2)
        
        # Status
        y_offset = 190
        status_text = "ACTIVE" if recognizing else "PAUSED"
        status_color = (0, 255, 0) if recognizing else (100, 100, 100)
        draw_text_with_bg(display, f"Status: {status_text}", (ui_x + margin, y_offset), 
                         scale=1.0, color=status_color, thickness=2)
        
        # Hand detection
        y_offset += 55
        right_text = "Right Hand: DETECTED" if right_hand else "Right Hand: ---"
        right_color = (0, 255, 0) if right_hand else (100, 100, 100)
        draw_text_with_bg(display, right_text, (ui_x + margin, y_offset), 
                         scale=0.8, color=right_color, thickness=2)
        
        y_offset += 40
        if left_hand:
            if index_pinch:
                left_text = "Left: INDEX (Add)"
                left_color = (0, 255, 100)
            elif middle_pinch:
                left_text = "Left: MIDDLE (Backspace)"
                left_color = (255, 200, 100)
            elif pinky_pinch:
                left_text = "Left: PINKY (Reset)"
                left_color = (100, 100, 255)
            else:
                left_text = "Left Hand: READY"
                left_color = (0, 255, 0)
        else:
            left_text = "Left Hand: ---"
            left_color = (100, 100, 100)
        draw_text_with_bg(display, left_text, (ui_x + margin, y_offset), 
                         scale=0.8, color=left_color, thickness=2)
        
        # FPS and inference
        y_offset += 45
        avg_inference = np.mean(inference_times) if inference_times else 0
        draw_text_with_bg(display, f"FPS: {fps:.1f}  |  Inference: {avg_inference:.1f}ms", 
                         (ui_x + margin, y_offset), scale=0.7, color=(180, 180, 180), thickness=1)
        
        # ====================================================================
        # TOP PREDICTIONS
        # ====================================================================
        y_offset += 60
        cv2.line(display, (ui_x + margin, y_offset - 15), (DISPLAY_WIDTH - margin, y_offset - 15), (60, 60, 70), 2)
        draw_text_with_bg(display, "PREDICTIONS", (ui_x + margin, y_offset + 10), 
                         scale=1.0, color=(255, 255, 0), thickness=2)
        
        y_offset += 60
        if current_pred and recognizing:
            for i, pred in enumerate(current_pred):
                conf = pred['confidence']
                class_name = pred['class']
                
                # Color coding
                if i == 0:
                    color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (255, 200, 100)
                    scale = 2.0
                    thickness = 3
                else:
                    color = (180, 180, 180)
                    scale = 0.9
                    thickness = 2
                
                # Class name
                draw_text_with_bg(display, f"{i+1}. {class_name}", 
                                 (ui_x + margin, y_offset), 
                                 scale=scale, color=color, thickness=thickness)
                
                # Confidence percentage on same line
                conf_text = f"{conf*100:.0f}%"
                draw_text_with_bg(display, conf_text, 
                                 (ui_x + UI_WIDTH - 120, y_offset), 
                                 scale=scale * 0.5, color=color, thickness=2)
                
                # Confidence bar below
                bar_y = y_offset + 15
                bar_width = UI_WIDTH - margin * 2
                bar_height = 25 if i == 0 else 12
                draw_confidence_bar(display, ui_x + margin, bar_y, 
                                   bar_width, bar_height, conf, color=color)
                
                y_offset += 80 if i == 0 else 50
        else:
            if recognizing:
                draw_text_with_bg(display, "Show RIGHT hand", 
                                 (ui_x + margin, y_offset), 
                                 scale=0.9, color=(150, 150, 150), thickness=2)
                draw_text_with_bg(display, "for letter prediction", 
                                 (ui_x + margin, y_offset + 35), 
                                 scale=0.7, color=(120, 120, 120), thickness=1)
            else:
                draw_text_with_bg(display, "Press SPACE to start", 
                                 (ui_x + margin, y_offset), 
                                 scale=0.9, color=(150, 150, 150), thickness=2)
        
        # ====================================================================
        # PINCH GESTURES INDICATOR
        # ====================================================================
        y_offset += 50
        cv2.line(display, (ui_x + margin, y_offset - 15), (DISPLAY_WIDTH - margin, y_offset - 15), (60, 60, 70), 2)
        draw_text_with_bg(display, "GESTURES", (ui_x + margin, y_offset + 10), 
                         scale=1.0, color=(255, 255, 0), thickness=2)
        
        y_offset += 45
        if left_hand:
            # Index pinch (add letter)
            index_progress = pinch_trigger.index_count / PINCH_FRAMES_REQUIRED
            index_progress = min(index_progress, 1.0)
            index_color = (0, 255, 100) if index_pinch else (80, 80, 80)
            draw_text_with_bg(display, "ADD (Index):", (ui_x + margin, y_offset), 
                             scale=0.6, color=index_color, thickness=2)
            draw_confidence_bar(display, ui_x + margin, y_offset + 10, 
                               UI_WIDTH - margin * 2, 16, index_progress, color=index_color)
            
            y_offset += 45
            
            # Middle pinch (backspace)
            middle_progress = pinch_trigger.middle_count / PINCH_FRAMES_REQUIRED
            middle_progress = min(middle_progress, 1.0)
            middle_color = (255, 200, 100) if middle_pinch else (80, 80, 80)
            draw_text_with_bg(display, "BACKSPACE (Middle):", (ui_x + margin, y_offset), 
                             scale=0.6, color=middle_color, thickness=2)
            draw_confidence_bar(display, ui_x + margin, y_offset + 10, 
                               UI_WIDTH - margin * 2, 16, middle_progress, color=middle_color)
            
            y_offset += 45
            
            # Pinky pinch (reset)
            pinky_progress = pinch_trigger.pinky_count / PINCH_FRAMES_REQUIRED
            pinky_progress = min(pinky_progress, 1.0)
            pinky_color = (100, 100, 255) if pinky_pinch else (80, 80, 80)
            draw_text_with_bg(display, "RESET (Pinky):", (ui_x + margin, y_offset), 
                             scale=0.6, color=pinky_color, thickness=2)
            draw_confidence_bar(display, ui_x + margin, y_offset + 10, 
                               UI_WIDTH - margin * 2, 16, pinky_progress, color=pinky_color)
        else:
            draw_text_with_bg(display, "Show LEFT hand for gestures", (ui_x + margin, y_offset), 
                             scale=0.7, color=(100, 100, 100), thickness=1)
        
        # Recent addition feedback
        if last_added and time.time() - last_added_time < 1.0:
            y_offset += 80
            draw_text_with_bg(display, f"ADDED: {last_added}", (ui_x + margin, y_offset), 
                             scale=1.2, color=(0, 255, 100), thickness=3)
        
        # ====================================================================
        # TYPED TEXT
        # ====================================================================
        y_offset = DISPLAY_HEIGHT - 280
        cv2.line(display, (ui_x + margin, y_offset - 15), (DISPLAY_WIDTH - margin, y_offset - 15), (60, 60, 70), 2)
        draw_text_with_bg(display, "TYPED TEXT", (ui_x + margin, y_offset + 10), 
                         scale=1.0, color=(255, 255, 0), thickness=2)
        
        y_offset += 50
        text_box_height = 120
        cv2.rectangle(display, (ui_x + margin - 5, y_offset - 10), 
                     (DISPLAY_WIDTH - margin + 5, y_offset + text_box_height), 
                     (40, 40, 45), -1)
        cv2.rectangle(display, (ui_x + margin - 5, y_offset - 10), 
                     (DISPLAY_WIDTH - margin + 5, y_offset + text_box_height), 
                     (80, 80, 90), 3)
        
        if typed_text:
            max_chars = 18
            lines = [typed_text[i:i+max_chars] for i in range(0, len(typed_text), max_chars)]
            for i, line in enumerate(lines[-3:]):  # Show last 3 lines
                draw_text_with_bg(display, line, (ui_x + margin + 10, y_offset + 25 + i*35), 
                                 scale=1.0, color=(255, 255, 255), thickness=2)
        else:
            draw_text_with_bg(display, "(make fist to add...)", (ui_x + margin + 10, y_offset + 40), 
                             scale=0.8, color=(100, 100, 100), thickness=1)
        
        # ====================================================================
        # CONTROLS
        # ====================================================================
        y_offset = DISPLAY_HEIGHT - 80
        draw_text_with_bg(display, "SPACE: Start/Pause   V: Viz   C: Clear   Q: Quit", 
                         (ui_x + margin, y_offset), 
                         scale=0.6, color=(140, 140, 150), thickness=1)
        
        # ====================================================================
        # LARGE TYPED TEXT OVERLAY (center bottom of camera view)
        # ====================================================================
        cam_width = DISPLAY_WIDTH - UI_WIDTH
        if typed_text:
            display_text = typed_text[-20:]  # Show last 20 chars
            # Calculate text size for centering
            font_scale = 2.5
            thickness = 4
            (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position: center horizontally in camera area, near bottom
            text_x = (cam_width - text_w) // 2
            text_y = DISPLAY_HEIGHT - 80
            
            # Draw background box
            padding = 20
            cv2.rectangle(display, 
                         (text_x - padding, text_y - text_h - padding),
                         (text_x + text_w + padding, text_y + baseline + padding),
                         (0, 0, 0), -1)
            cv2.rectangle(display, 
                         (text_x - padding, text_y - text_h - padding),
                         (text_x + text_w + padding, text_y + baseline + padding),
                         (0, 255, 255), 3)
            
            # Draw text
            cv2.putText(display, display_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # ====================================================================
        # SHOW FULLSCREEN
        # ====================================================================
        cv2.namedWindow("ASL Spelling Mode", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("ASL Spelling Mode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("ASL Spelling Mode", display)
        
        # Update FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            recognizing = not recognizing
            if recognizing:
                smoother.clear()
                pinch_trigger.clear()
                print("▶️  Recognition started")
            else:
                print("⏸️  Recognition paused")
        
        elif key == ord('v') or key == ord('V'):
            viz_idx = (viz_idx + 1) % len(viz_modes)
            viz_mode = viz_modes[viz_idx]
            print(f"🔄 Visualization: {viz_mode.upper()}")
        
        elif key == ord('c') or key == ord('C'):
            typed_text = ""
            smoother.clear()
            pinch_trigger.clear()
            print("🗑️  Cleared all")
        
        elif key == 8:  # Backspace
            typed_text = typed_text[:-1] if typed_text else ""
            print(f"⌫ Backspace -> '{typed_text}'")
        
        elif key == ord('q') or key == ord('Q') or key == 27:
            print("👋 Exiting...")
            break
    
    # Cleanup
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    
    print(f"\n📝 Final text: '{typed_text}'")
    print("✅ Done!")


if __name__ == "__main__":
    main()
