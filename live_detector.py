import os
import sys

# Silence system backend logs cleanly
os.putenv('TF_CPP_MIN_LOG_LEVEL', '3')
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pyttsx3
import threading

# Force-close any stuck background camera windows
cv2.destroyAllWindows()

# =====================================================================
# 1. TEXT-TO-SPEECH (VOICE SYSTEM)
# =====================================================================
engine = pyttsx3.init()
engine.setProperty('rate', 140)

def speak_async(text):
    def speech_worker(t):
        try:
            engine.say(t)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=speech_worker, args=(text,), daemon=True).start()

# =====================================================================
# 2. PATH SETUP & INTERPRETER INITIALIZATION (Bypasses Keras Errors)
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print(f"\n[ERROR] Missing files! Put '{MODEL_PATH}' and '{LABELS_PATH}' in: {os.getcwd()}")
    sys.exit()

print("[INFO] Initializing stable TFLite Runtime Engine...")
import tensorflow as tf

# Load the TFLite model and allocate structural tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get structural input and output details for data formatting mapping
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Parse the labels index map
class_names = []
with open(LABELS_PATH, "r") as f:
    for line in f.readlines():
        parts = line.strip().split(" ", 1)
        label = parts[1] if len(parts) > 1 else parts[0]
        class_names.append(label)

print("[SUCCESS] Connected to TFLite Model Engine! Target classes:", class_names)

# =====================================================================
# 3. LIVE WEBCAM VIDEO STREAM LOOP
# =====================================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Webcam could not open. Verify privacy permissions.")
    sys.exit()

last_announced = ""
print("\n>>> DETECTOR RUNNING: Hold a note up to the camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # Pre-process live matrix pixels to match 224x224 RGB expectations
    input_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    input_image = np.asarray(input_image, dtype=np.float32).reshape(1, 224, 224, 3)
    input_image = (input_image / 127.5) - 1

    # Feed input data to the TFLite engine interpreter
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Retrieve statistical analysis outputs from model tracking nodes
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    highest_index = np.argmax(prediction)
    detected_label = class_names[highest_index]
    confidence_score = prediction[highest_index]

    # Voice Output and Graphic UI Logic
    if confidence_score > 0.85 and "background" not in detected_label.lower():
        clean_name = detected_label.replace('_', ' ')
        ui_string = f"{clean_name} ({confidence_score * 100:.0f}%)"
        
        cv2.putText(display_frame, ui_string, (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        if detected_label != last_announced:
            print(f"[MATCHED] Target: {clean_name} | Confidence: {confidence_score:.2%}")
            speak_async(f"This is a {clean_name} note")
            last_announced = detected_label
    else:
        cv2.putText(display_frame, "Scanning for currency...", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        last_announced = ""

    cv2.imshow("AI Assistive Currency Recognition System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] App shut down cleanly.")

