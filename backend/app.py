"""
Flask Backend for CAIretaker - Real-time Fall Detection with Multi-Person Tracking
Integrates YOLOv8 pose detection with bounding boxes and tracking IDs
"""

from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import time
from collections import deque, defaultdict
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    OUTPUT_DIR = "backend/spatial_branch_output"
    CNN_MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn1d_model.pth")
    YOLO_MODEL = "yolo11m-pose.pt"
    CONFIDENCE_THRESHOLD = 0.5
    SMOOTHING_WINDOW = 5
    CLASS_NAMES = {0: "Standing", 1: "Sitting", 2: "Fallen"}
    CLASS_COLORS = {
        0: (0, 255, 0),    # Standing - Green
        1: (255, 255, 0),  # Sitting - Yellow
        2: (0, 0, 255)     # Fallen - Red
    }
    NUM_KEYPOINTS = 17
    NUM_COORDS = 3
    # Minimum confidence for keypoints to be considered valid
    KEYPOINT_MIN_CONFIDENCE = 0.3
    # Minimum number of keypoints required for classification
    MIN_KEYPOINTS_REQUIRED = 10
    # Critical keypoints that should be visible (hips, shoulders, knees)
    CRITICAL_KEYPOINTS = [5, 6, 11, 12, 13, 14]  # shoulders, hips, knees


# 1D-CNN Model Architecture
class Spatial1DCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=3, dropout_rate=0.3):
        super(Spatial1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.relu_fc2(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x


class FallDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO pose model with tracking enabled
        print("Loading YOLO pose model with tracking...")
        self.pose_model = YOLO(Config.YOLO_MODEL)
        print("✓ YOLO model loaded")
        
        # Load 1D-CNN classifier
        print("Loading 1D-CNN classifier...")
        self.cnn_model = Spatial1DCNN(
            input_channels=Config.NUM_COORDS,
            num_classes=len(Config.CLASS_NAMES)
        ).to(self.device)
        
        if os.path.exists(Config.CNN_MODEL_PATH):
            checkpoint = torch.load(Config.CNN_MODEL_PATH, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ 1D-CNN model loaded (epoch: {checkpoint.get('epoch', 'N/A')}, val_acc: {checkpoint.get('val_acc', 0):.2f}%)")
            else:
                self.cnn_model.load_state_dict(checkpoint)
                print("✓ 1D-CNN model loaded (legacy format)")
            
            self.cnn_model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {Config.CNN_MODEL_PATH}")
        
        # Prediction smoothing per track ID
        self.prediction_buffers = defaultdict(lambda: deque(maxlen=Config.SMOOTHING_WINDOW))
        
    def check_keypoints_validity(self, keypoints):
        """Check if keypoints are sufficient for classification"""
        if len(keypoints) == 0:
            return False, "No keypoints detected"
        
        # Count valid keypoints
        valid_keypoints = np.sum(keypoints[:, 2] > Config.KEYPOINT_MIN_CONFIDENCE)
        if valid_keypoints < Config.MIN_KEYPOINTS_REQUIRED:
            return False, f"Insufficient keypoints ({valid_keypoints}/{Config.MIN_KEYPOINTS_REQUIRED})"
        
        # Check critical keypoints for sitting detection
        # For sitting: we especially need hips (11,12) and knees (13,14) visible
        critical_valid = sum(1 for idx in Config.CRITICAL_KEYPOINTS 
                           if idx < len(keypoints) and keypoints[idx, 2] > Config.KEYPOINT_MIN_CONFIDENCE)
        if critical_valid < len(Config.CRITICAL_KEYPOINTS) * 0.5:  # Reduced to 50% threshold
            return False, f"Critical keypoints missing ({critical_valid}/{len(Config.CRITICAL_KEYPOINTS)})"
        
        return True, "Valid"
    
    def get_pose_features(self, keypoints):
        """Extract pose features for debugging and analysis"""
        features = {}
        
        # Hip positions (11=left hip, 12=right hip)
        if keypoints[11, 2] > 0.3 and keypoints[12, 2] > 0.3:
            hip_y = (keypoints[11, 1] + keypoints[12, 1]) / 2
            features['hip_y'] = hip_y
        
        # Knee positions (13=left knee, 14=right knee)
        if keypoints[13, 2] > 0.3 and keypoints[14, 2] > 0.3:
            knee_y = (keypoints[13, 1] + keypoints[14, 1]) / 2
            features['knee_y'] = knee_y
        
        # Shoulder positions (5=left shoulder, 6=right shoulder)
        if keypoints[5, 2] > 0.3 and keypoints[6, 2] > 0.3:
            shoulder_y = (keypoints[5, 1] + keypoints[6, 1]) / 2
            features['shoulder_y'] = shoulder_y
        
        # Calculate torso angle (indicator of sitting vs standing)
        if 'shoulder_y' in features and 'hip_y' in features:
            features['torso_height'] = abs(features['hip_y'] - features['shoulder_y'])
        
        # Calculate knee bend (sitting typically has bent knees)
        if 'hip_y' in features and 'knee_y' in features:
            features['hip_knee_distance'] = abs(features['knee_y'] - features['hip_y'])
        
        return features
    
    def detect(self, frame):
        """Detect pose and classify activity with tracking"""
        # Run YOLO pose detection with tracking
        results = self.pose_model.track(frame, persist=True, verbose=False, conf=0.5)
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes
                
                # Get tracking IDs if available
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
                
                for idx, keypoints in enumerate(keypoints_data):
                    # Get bounding box
                    box = boxes.xyxy[idx].cpu().numpy()
                    box_conf = boxes.conf[idx].cpu().numpy()
                    
                    # Get track ID or use index
                    track_id = int(track_ids[idx]) if track_ids is not None else idx
                    
                    # Check if keypoints are valid for classification
                    is_valid, validity_reason = self.check_keypoints_validity(keypoints)
                    
                    if is_valid and box_conf > 0.4:
                        # Prepare input for CNN
                        keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
                        
                        # Classify pose
                        with torch.no_grad():
                            outputs = self.cnn_model(keypoints_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence_val, predicted = torch.max(probabilities, 1)
                            
                            prediction = predicted.item()
                            confidence = confidence_val.item()
                        
                        # Smooth predictions per track ID
                        self.prediction_buffers[track_id].append(prediction)
                        if len(self.prediction_buffers[track_id]) >= Config.SMOOTHING_WINDOW // 2:
                            prediction = max(set(self.prediction_buffers[track_id]), 
                                           key=self.prediction_buffers[track_id].count)
                        
                        status = "classified"
                    else:
                        # Don't classify, just show tracking
                        prediction = None
                        confidence = 0.0
                        status = "tracking_only"
                        validity_reason = validity_reason if not is_valid else "Low box confidence"
                    
                    detections.append({
                        'track_id': track_id,
                        'box': box,
                        'box_conf': float(box_conf),
                        'keypoints': keypoints,
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'status': status,
                        'reason': validity_reason if status == "tracking_only" else None
                    })
        
        return detections
    
    def draw_results(self, frame, detections):
        """Draw bounding boxes, IDs, and classifications on frame"""
        for detection in detections:
            track_id = detection['track_id']
            box = detection['box']
            keypoints = detection['keypoints']
            prediction = detection['prediction']
            confidence = detection['confidence']
            status = detection['status']
            
            # Determine color based on prediction or default for tracking only
            if status == "classified" and prediction is not None:
                color = Config.CLASS_COLORS.get(prediction, (255, 255, 255))
                class_name = Config.CLASS_NAMES.get(prediction, "Unknown")
            else:
                color = (128, 128, 128)  # Gray for tracking only
                class_name = "Tracking"
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw keypoints (if valid)
            if status == "classified":
                for i, kp in enumerate(keypoints):
                    x, y, conf = kp
                    if conf > Config.KEYPOINT_MIN_CONFIDENCE:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                
                # Draw skeleton connections
                connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                
                for start, end in connections:
                    if (keypoints[start, 2] > Config.KEYPOINT_MIN_CONFIDENCE and 
                        keypoints[end, 2] > Config.KEYPOINT_MIN_CONFIDENCE):
                        pt1 = (int(keypoints[start, 0]), int(keypoints[start, 1]))
                        pt2 = (int(keypoints[end, 0]), int(keypoints[end, 1]))
                        cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw ID and status label above box
            label_y = max(y1 - 10, 20)
            
            if status == "classified":
                label = f"ID:{track_id} | {class_name} ({confidence*100:.0f}%)"
            else:
                label = f"ID:{track_id} | {class_name}"
            
            # Get label size for background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for label
            cv2.rectangle(frame, (x1, label_y - label_h - 8), (x1 + label_w + 10, label_y + 2), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


# Global variables
detector = None
camera = None
camera_lock = __import__('threading').Lock()
camera_initialized = False
current_status = {
    'people_detected': 0,
    'detections': [],
    'fps': 0
}
status_lock = __import__('threading').Lock()


def initialize_detector():
    """Initialize the fall detector"""
    global detector
    try:
        detector = FallDetector()
        print("✓ Fall detector initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing detector: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_camera():
    """Initialize camera with multiple attempts"""
    global camera, camera_initialized
    
    with camera_lock:
        if camera_initialized and camera is not None and camera.isOpened():
            return True
        
        print("\n" + "="*60)
        print("Initializing Camera...")
        print("="*60)
        
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
        
        for cam_index in [0, 1, 2]:
            print(f"Trying camera index {cam_index}...")
            try:
                cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                
                if not cap.isOpened():
                    cap = cv2.VideoCapture(cam_index)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ Camera {cam_index} opened successfully!")
                        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        camera = cap
                        camera_initialized = True
                        print("="*60 + "\n")
                        return True
                    else:
                        print(f"  Camera {cam_index} opened but couldn't read frame")
                        cap.release()
                else:
                    print(f"  Camera {cam_index} not available")
                    
            except Exception as e:
                print(f"  Error with camera {cam_index}: {e}")
        
        print("✗ Failed to initialize any camera")
        print("="*60 + "\n")
        return False


def generate_frames():
    """Generate frames with pose detection and tracking"""
    global current_status, camera
    
    if not initialize_camera():
        print("✗ Camera initialization failed, sending error frame")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Not Available", (120, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(error_frame, "Please check camera connection", (80, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)
    
    print("✓ Starting frame generation with multi-person tracking...")
    frame_count = 0
    fps_time = time.time()
    fps_value = 0
    consecutive_failures = 0
    
    while True:
        try:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    print("⚠ Camera lost, attempting to reconnect...")
                    if not initialize_camera():
                        time.sleep(1)
                        continue
                
                success, frame = camera.read()
            
            if not success or frame is None:
                consecutive_failures += 1
                print(f"⚠ Frame read failed (attempt {consecutive_failures})")
                
                if consecutive_failures > 10:
                    print("✗ Too many consecutive failures, reinitializing camera...")
                    camera_initialized = False
                    if not initialize_camera():
                        time.sleep(1)
                    consecutive_failures = 0
                
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            
            # Detect and classify with tracking
            if detector is not None:
                detections = detector.detect(frame)
                
                # Draw results
                frame = detector.draw_results(frame, detections)
                
                # Update status
                with status_lock:
                    current_status['people_detected'] = len(detections)
                    current_status['detections'] = [
                        {
                            'id': d['track_id'],
                            'status': Config.CLASS_NAMES.get(d['prediction'], 'Tracking') if d['status'] == 'classified' else 'Tracking',
                            'confidence': d['confidence'],
                            'is_fall': d['prediction'] == 2 and d['confidence'] > Config.CONFIDENCE_THRESHOLD if d['status'] == 'classified' else False
                        }
                        for d in detections
                    ]
                    
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    current_time = time.time()
                    fps_value = 10 / (current_time - fps_time)
                    fps_time = current_time
                    current_status['fps'] = round(fps_value, 1)
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps_value:.1f}", 
                           (frame.shape[1] - 200, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Draw people count
                cv2.putText(frame, f"People: {len(detections)}", 
                           (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print("⚠ Frame encoding failed")
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"✗ Error in frame generation: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    print("📹 Video feed requested")
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def get_status():
    """Get current detection status"""
    with status_lock:
        return jsonify(current_status)


@app.route('/health')
def health():
    """Health check endpoint"""
    with camera_lock:
        cam_available = camera is not None and camera.isOpened()
    
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None,
        'camera_available': cam_available
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CAIretaker Backend - Multi-Person Fall Detection with Tracking")
    print("="*60)
    
    if initialize_detector():
        print("\nStarting Flask server...")
        print("Backend will be available at: http://localhost:5000")
        print("Video feed: http://localhost:5000/video_feed")
        print("Status API: http://localhost:5000/status")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("\n✗ Failed to initialize. Please check model files.")