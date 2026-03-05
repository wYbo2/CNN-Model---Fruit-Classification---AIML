import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.keras.mixed_precision.set_global_policy('float32')

MODEL_PATH = 'models/M3.keras'
IMG_SIZE = (224, 224)
class_names = ['banana', 'dragonfruit', 'unknown']

print(f"Loading model from '{MODEL_PATH}'...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nWebcam started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_img = cv2.resize(frame, IMG_SIZE)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) 
    input_img = np.expand_dims(input_img, axis=0)

    # Use the same prediction method as phone_cam.py
    # 'training=False' is faster and more consistent than .predict()
    predictions = model(input_img, training=False)
    raw_score = predictions.numpy()[0]

    # --- THE FIX: SMART SOFTMAX CHECK ---
    # Only apply softmax if the model outputs "logits" (numbers outside 0-1)
    # This matches your phone_cam.py logic exactly.
    if np.max(raw_score) > 1.0 or np.min(raw_score) < 0:
        score = tf.nn.softmax(raw_score).numpy()
    else:
        score = raw_score # Don't apply softmax twice!

    class_index = np.argmax(score)
    confidence = 99 * np.max(score)
    predicted_label = class_names[class_index]

    if predicted_label == 'unknown':
        box_color = (0, 0, 255)  
    else:
        box_color = (0, 255, 0) 

    cv2.rectangle(frame, (0, 0), (400, 60), box_color, -1)
    text = f"{predicted_label}: {confidence:.1f}%"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Simple Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()