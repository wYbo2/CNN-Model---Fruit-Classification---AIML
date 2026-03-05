import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from threading import Thread, Lock

tf.keras.mixed_precision.set_global_policy('float32')

MODEL_PATH = 'models/M3.keras'
IP_WEBCAM_URL = "http://172.23.58.243:8080/video" 
IMG_SIZE = (224, 224)
class_names = ['banana', 'dragonfruit', 'unknown'] 


MIN_BRIGHTNESS = 20.0   
MIN_VARIANCE = 5.0     

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            
            grabbed, frame = self.stream.read()
            
            if grabbed:
                self.grabbed = grabbed
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class AIWorker:
    def __init__(self, model_path):
        print(f"Loading model from '{model_path}'...")
        try:
            self.model = load_model(model_path, compile=False)
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            self.model(dummy, training=False)
            print("✅ Model loaded and warmed up!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()
            
        self.frame_to_process = None
        self.latest_result = ("Initializing...", 0.0, (100,100,100)) 
        self.debug_stats = "Init..."
        self.stopped = False
        self.lock = Lock()
        self.new_frame_available = False

    def start(self):
        Thread(target=self.run, args=(), daemon=True).start()
        return self

    def predict(self, frame):
        with self.lock:
            self.frame_to_process = frame.copy()
            self.new_frame_available = True

    def get_result(self):
        with self.lock:
            return self.latest_result, self.debug_stats

    def run(self):
        while not self.stopped:
            process_now = False
            with self.lock:
                if self.new_frame_available:
                    img_for_net = self.frame_to_process
                    self.new_frame_available = False
                    process_now = True
            
            if process_now:
                gray = cv2.cvtColor(img_for_net, cv2.COLOR_BGR2GRAY)
                mean, std_dev = cv2.meanStdDev(gray)
                brightness = mean[0][0]
                variance = std_dev[0][0]
                
                stats_text = f"Brit: {brightness:.1f} | Var: {variance:.1f}"

                if brightness < MIN_BRIGHTNESS:
                    result = ("Too Dark", 0.0, (50, 50, 50))
                elif variance < MIN_VARIANCE:
                    result = ("Empty / Blur", 0.0, (50, 50, 50))
                else:
                    try:
                        input_img = cv2.resize(img_for_net, IMG_SIZE)
                        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                        input_img = input_img.astype(np.float32) 
                        input_img = np.expand_dims(input_img, axis=0)

                        predictions = self.model(input_img, training=False)
                        raw_score = predictions.numpy()[0]

                        if np.max(raw_score) > 1.0 or np.min(raw_score) < 0:
                            curr_score = tf.nn.softmax(raw_score).numpy()
                        else:
                            curr_score = raw_score

                        class_index = np.argmax(curr_score)
                        confidence = float(np.max(curr_score) * 100)
                        label = class_names[class_index]

                        if label == "unknown":
                            color = (0, 0, 255) # Red
                            label_text = "Unknown"
                        elif confidence < 60:
                            color = (100, 100, 100) # Grey
                            label_text = "Uncertain"
                        else:
                            color = (0, 255, 0) # Green
                            label_text = label
                            
                        result = (label_text, confidence, color)
                    except Exception as e:
                        print(f"AI Error: {e}")
                        result = ("Error", 0.0, (0,0,255))
                
                with self.lock:
                    self.latest_result = result
                    self.debug_stats = stats_text
            
            else:
                time.sleep(0.01)

    def stop(self):
        self.stopped = True


ai_worker = AIWorker(MODEL_PATH).start()

print(f"Connecting to camera at {IP_WEBCAM_URL}...")
try:
    video_stream = VideoStream(IP_WEBCAM_URL).start()
    time.sleep(1.0)
    if video_stream.frame is None: raise Exception("No frame")
except Exception as e:
    print(f"❌ Connection Error: {e}")
    ai_worker.stop()
    exit()

print("\n🚀 System Running! Press 'q' to quit.")

while True:
    frame = video_stream.read()
    if frame is None: break

    ai_worker.predict(frame)

    (label, conf, color), debug_stats = ai_worker.get_result()

    height, width = frame.shape[:2]
    new_height = int(800 * height / width)
    display_frame = cv2.resize(frame, (800, new_height))

    cv2.rectangle(display_frame, (0, 0), (350, 85), color, -1)
    cv2.putText(display_frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if label not in ["Too Dark", "Empty / Blur", "Initializing..."]:
        cv2.putText(display_frame, f"Conf: {conf:.1f}%", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(display_frame, debug_stats, (10, new_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Smooth AI Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.stop()
ai_worker.stop()
cv2.destroyAllWindows()