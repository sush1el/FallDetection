"""
Flask Backend for CAIretaker - Enhanced Fall Detection with Multi-Person Tracking
Exact same detection logic as inference code + person tracking with IDs and database logging
"""

from flask import Flask, Response, jsonify, request
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
from database import get_database

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    OUTPUT_DIR = "backend\spatial_branch_output"
    CNN_MODEL_PATH = os.path.join(OUTPUT_DIR, "enhanced_cnn1d_model.pth")
    YOLO_MODEL = "yolo11m-pose.pt"
    CONFIDENCE_THRESHOLD = 0.65
    SMOOTHING_WINDOW = 7
    CLASS_NAMES = {0: "Standing", 1: "Sitting", 2: "Fallen"}
    CLASS_COLORS = {
        0: (0, 255, 0),    # Standing - Green
        1: (255, 255, 0),  # Sitting - Yellow
        2: (0, 0, 255)     # Fallen - Red
    }
    NUM_KEYPOINTS = 17
    NUM_COORDS = 3
    NUM_SPATIAL_FEATURES = 7
    LOCATION_NAME = "UAC - Exhibit"


# ============================================================================
# ENHANCED MODEL ARCHITECTURE
# ============================================================================

class EnhancedSpatial1DCNN(nn.Module):
    """Dual-branch architecture: CNN for keypoints + MLP for spatial features"""
    def __init__(self, num_classes=3, dropout_rate=0.4):
        super(EnhancedSpatial1DCNN, self).__init__()
        
        # Branch 1: CNN for keypoint sequences
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Branch 2: MLP for spatial features
        self.spatial_fc1 = nn.Linear(7, 32)
        self.spatial_bn1 = nn.BatchNorm1d(32)
        self.spatial_relu1 = nn.ReLU()
        self.spatial_dropout1 = nn.Dropout(0.3)
        
        self.spatial_fc2 = nn.Linear(32, 64)
        self.spatial_bn2 = nn.BatchNorm1d(64)
        self.spatial_relu2 = nn.ReLU()
        self.spatial_dropout2 = nn.Dropout(0.3)
        
        # Fusion layers
        self.fusion_fc1 = nn.Linear(320, 128)
        self.fusion_bn1 = nn.BatchNorm1d(128)
        self.fusion_relu1 = nn.ReLU()
        self.fusion_dropout1 = nn.Dropout(dropout_rate)
        
        self.fusion_fc2 = nn.Linear(128, 64)
        self.fusion_bn2 = nn.BatchNorm1d(64)
        self.fusion_relu2 = nn.ReLU()
        self.fusion_dropout2 = nn.Dropout(dropout_rate)
        
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, keypoints, spatial_features):
        # Branch 1: CNN
        x = keypoints.permute(0, 2, 1)
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        x = self.global_pool(x).squeeze(-1)
        
        # Branch 2: MLP
        s = self.spatial_dropout1(self.spatial_relu1(self.spatial_bn1(self.spatial_fc1(spatial_features))))
        s = self.spatial_dropout2(self.spatial_relu2(self.spatial_bn2(self.spatial_fc2(s))))
        
        # Fusion
        combined = torch.cat([x, s], dim=1)
        combined = self.fusion_dropout1(self.fusion_relu1(self.fusion_bn1(self.fusion_fc1(combined))))
        combined = self.fusion_dropout2(self.fusion_relu2(self.fusion_bn2(self.fusion_fc2(combined))))
        
        output = self.fc_out(combined)
        return output


# ============================================================================
# SPATIAL FEATURE EXTRACTION (same as inference)
# ============================================================================

def calculate_body_angle(keypoints):
    """Calculate angle of body from vertical axis"""
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if (left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3 or 
        left_hip[2] < 0.3 or right_hip[2] < 0.3):
        return None
    
    shoulder_center = np.array([
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ])
    hip_center = np.array([
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    ])
    
    body_vector = hip_center - shoulder_center
    vertical_vector = np.array([0, 1])
    
    if np.linalg.norm(body_vector) < 1e-6:
        return None
    
    cos_angle = np.dot(body_vector, vertical_vector) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def check_hip_position(keypoints, image_shape):
    """Check if hips are at ground level"""
    h, w = image_shape[:2]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    hip_y_values = []
    if left_hip[2] > 0.3:
        hip_y_values.append(left_hip[1])
    if right_hip[2] > 0.3:
        hip_y_values.append(right_hip[1])
    
    if not hip_y_values:
        return None
    
    return np.mean(hip_y_values) / h


def check_knee_position(keypoints, image_shape):
    """Check knee positions"""
    h, w = image_shape[:2]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    visible_knees = []
    visible_hips = []
    
    if left_knee[2] > 0.3:
        visible_knees.append(left_knee)
    if right_knee[2] > 0.3:
        visible_knees.append(right_knee)
    if left_hip[2] > 0.3:
        visible_hips.append(left_hip)
    if right_hip[2] > 0.3:
        visible_hips.append(right_hip)
    
    if not visible_knees or not visible_hips:
        return None, None
    
    avg_knee_y = np.mean([k[1] for k in visible_knees])
    avg_hip_y = np.mean([h[1] for h in visible_hips])
    
    knee_hip_distance = (avg_knee_y - avg_hip_y) / h
    relative_knee_y = avg_knee_y / h
    
    return knee_hip_distance, relative_knee_y


def calculate_center_of_mass(keypoints):
    """Calculate approximate center of mass Y-coordinate"""
    torso_indices = [5, 6, 11, 12]
    
    visible_torso = []
    for idx in torso_indices:
        if keypoints[idx, 2] > 0.3:
            visible_torso.append(keypoints[idx, 1])
    
    if len(visible_torso) < 2:
        return None
    
    return np.mean(visible_torso)


def calculate_body_dimensions(keypoints):
    """Calculate body width and height ratios - MUST MATCH TRAINING CODE"""
    visible_kps = keypoints[keypoints[:, 2] > 0.3]
    
    if len(visible_kps) < 5:
        return None, None
    
    x_coords = visible_kps[:, 0]
    y_coords = visible_kps[:, 1]
    
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    if height < 1:
        return None, None
    
    aspect_ratio = width / height
    relative_height = height
    
    return aspect_ratio, relative_height




def extract_spatial_features(keypoints, image_shape):
    """Extract 7 spatial features for the MLP branch - MUST MATCH TRAINING CODE EXACTLY"""
    # Handle both (h, w) and (h, w, c) formats
    if len(image_shape) == 3:
        h, w, _ = image_shape
    else:
        h, w = image_shape[:2]
    
    spatial_features = []
    
    # 1. Body angle (default: 45.0 - NOT 0.0!)
    angle = calculate_body_angle(keypoints)
    spatial_features.append(angle if angle is not None else 45.0)
    
    # 2. Hip height (default: 0.5)
    hip_height = check_hip_position(keypoints, image_shape)
    spatial_features.append(hip_height if hip_height is not None else 0.5)
    
    # 3. Aspect ratio (default: 0.5)
    aspect_ratio, body_height = calculate_body_dimensions(keypoints)
    spatial_features.append(aspect_ratio if aspect_ratio is not None else 0.5)
    
    # 4. Body height / h (default: 0.5)
    spatial_features.append(body_height / h if body_height else 0.5)
    
    # 5. Knee-hip distance (default: 0.0)
    knee_hip_dist, knee_height = check_knee_position(keypoints, image_shape)
    spatial_features.append(knee_hip_dist if knee_hip_dist is not None else 0.0)
    
    # 6. Knee height (default: 0.5)
    spatial_features.append(knee_height if knee_height is not None else 0.5)
    
    # 7. Center of mass / h (default: 0.5)
    com_y = calculate_center_of_mass(keypoints)
    spatial_features.append(com_y / h if com_y else 0.5)
    
    return np.array(spatial_features, dtype=np.float32)


# ============================================================================
# FALL DETECTOR CLASS
# ============================================================================

class FallDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO pose model with tracking enabled
        print("Loading YOLO pose model with tracking...")
        self.pose_model = YOLO(Config.YOLO_MODEL)
        print("✓ YOLO model loaded")
        
        # Load Enhanced 1D-CNN classifier
        print("Loading Enhanced 1D-CNN classifier...")
        self.cnn_model = EnhancedSpatial1DCNN(
            num_classes=len(Config.CLASS_NAMES),
            dropout_rate=0.4
        ).to(self.device)
        
        checkpoint = torch.load(Config.CNN_MODEL_PATH, map_location=self.device)
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model.eval()
        print(f"✓ Enhanced model loaded (epoch: {checkpoint.get('epoch', 'N/A')}, val_acc: {checkpoint.get('val_acc', 0):.2f}%)")
        
        # Prediction smoothing per track ID
        self.prediction_buffers = defaultdict(lambda: deque(maxlen=Config.SMOOTHING_WINDOW))
        
        # Track fall states per person
        self.fall_states = {}
        
        # Database instance
        self.db = get_database()
        
    def extract_features(self, keypoints, image_shape):
        """Extract normalized keypoint features (same as inference)"""
        h, w = image_shape[:2]
        
        normalized = keypoints.copy()
        normalized[:, 0] = normalized[:, 0] / w
        normalized[:, 1] = normalized[:, 1] / h
        
        return normalized
    
    def detect(self, frame):
        """Detect pose and classify activity with tracking (same logic as inference but multi-person)"""
        results = self.pose_model.track(frame, persist=True, verbose=False, conf=0.5)
        
        detections = []
        current_person_ids = set()
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes
                
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
                
                for idx, keypoints in enumerate(keypoints_data):
                    box = boxes.xyxy[idx].cpu().numpy()
                    box_conf = boxes.conf[idx].cpu().numpy()
                    track_id = int(track_ids[idx]) if track_ids is not None else idx
                    
                    current_person_ids.add(track_id)
                    
                    # Initialize fall state for new person
                    if track_id not in self.fall_states:
                        self.fall_states[track_id] = {
                            'is_fallen': False,
                            'incident_id': None
                        }
                    
                    # SAME VALIDATION AS INFERENCE: Only check if 10+ keypoints visible
                    visible_count = np.sum(keypoints[:, 2] > 0.3)
                    
                    if visible_count >= 10:
                        # Normalize keypoints
                        keypoints_normalized = self.extract_features(keypoints, frame.shape)
                        
                        # Prepare tensors
                        keypoints_tensor = torch.tensor(
                            keypoints_normalized.reshape(1, Config.NUM_KEYPOINTS, Config.NUM_COORDS),
                            dtype=torch.float32
                        ).to(self.device)
                        
                        # Extract spatial features
                        spatial_features = extract_spatial_features(keypoints, frame.shape)
                        spatial_tensor = torch.tensor(
                            spatial_features.reshape(1, Config.NUM_SPATIAL_FEATURES),
                            dtype=torch.float32
                        ).to(self.device)
                        
                        # Forward pass
                        with torch.no_grad():
                            outputs = self.cnn_model(keypoints_tensor, spatial_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence_val, predicted = torch.max(probabilities, 1)
                            
                            prediction = predicted.item()
                            confidence = confidence_val.item()
                        
                        # Smooth predictions per track ID
                        self.prediction_buffers[track_id].append(prediction)
                        if len(self.prediction_buffers[track_id]) >= Config.SMOOTHING_WINDOW // 2:
                            prediction = max(set(self.prediction_buffers[track_id]), 
                                           key=self.prediction_buffers[track_id].count)
                        
                        # Check for fall state changes
                        is_currently_fallen = (prediction == 2 and confidence > Config.CONFIDENCE_THRESHOLD)
                        was_fallen = self.fall_states[track_id]['is_fallen']
                        
                        # NEW FALL DETECTED
                        if is_currently_fallen and not was_fallen:
                            print(f"\n🚨 NEW FALL DETECTED: Person ID {track_id}")
                            incident_id = self.db.log_fall_incident(
                                person_id=track_id,
                                confidence=confidence,
                                location=Config.LOCATION_NAME
                            )
                            self.fall_states[track_id]['is_fallen'] = True
                            self.fall_states[track_id]['incident_id'] = incident_id
                        
                        # RECOVERY DETECTED
                        elif not is_currently_fallen and was_fallen:
                            print(f"\n✅ RECOVERY: Person ID {track_id} stood up")
                            self.db.resolve_fall_for_person(track_id)
                            self.fall_states[track_id]['is_fallen'] = False
                            self.fall_states[track_id]['incident_id'] = None
                        
                        status = "classified"
                    else:
                        # Not enough keypoints - just track
                        prediction = None
                        confidence = 0.0
                        status = "tracking_only"
                    
                    detections.append({
                        'track_id': track_id,
                        'box': box,
                        'box_conf': float(box_conf),
                        'keypoints': keypoints,
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'status': status,
                        'is_fallen': self.fall_states[track_id]['is_fallen'],
                        'incident_id': self.fall_states[track_id].get('incident_id'),
                        'visible_count': visible_count
                    })
        
        # Clean up tracking for people who left the frame
        disappeared_ids = set(self.fall_states.keys()) - current_person_ids
        for person_id in disappeared_ids:
            if self.fall_states[person_id]['is_fallen']:
                print(f"⚠ Person ID {person_id} with active fall left frame")
            del self.fall_states[person_id]
            if person_id in self.prediction_buffers:
                del self.prediction_buffers[person_id]
        
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
            is_fallen = detection.get('is_fallen', False)
            
            # Determine color and class name
            if is_fallen:
                color = (0, 0, 255)  # Red for active fall
                class_name = "FALLEN"
            elif status == "classified" and prediction is not None:
                color = Config.CLASS_COLORS.get(prediction, (255, 255, 255))
                class_name = Config.CLASS_NAMES.get(prediction, "Unknown")
            else:
                color = (128, 128, 128)
                class_name = "Tracking"
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            thickness = 5 if is_fallen else 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw keypoints
            if status == "classified":
                for i, kp in enumerate(keypoints):
                    x, y, conf = kp
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                
                connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                
                for start, end in connections:
                    if (keypoints[start, 2] > 0.3 and keypoints[end, 2] > 0.3):
                        pt1 = (int(keypoints[start, 0]), int(keypoints[start, 1]))
                        pt2 = (int(keypoints[end, 0]), int(keypoints[end, 1]))
                        cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw label
            label_y = max(y1 - 10, 20)
            
            if status == "tracking_only":
                if is_fallen:
                    label = f"ID:{track_id} | ⚠ FALLEN (Tracking) ⚠"
                else:
                    label = f"ID:{track_id} | Tracking ({detection['visible_count']}/10 kpts)"
            elif status == "classified":
                if is_fallen:
                    label = f"ID:{track_id} | ⚠ {class_name} ⚠"
                else:
                    label = f"ID:{track_id} | {class_name} ({confidence*100:.0f}%)"
            else:
                label = f"ID:{track_id} | Unknown"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, label_y - label_h - 8), (x1 + label_w + 10, label_y + 2), color, -1)
            cv2.putText(frame, label, (x1 + 5, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


# ============================================================================
# FLASK APPLICATION
# ============================================================================

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
    
    print("✓ Starting frame generation (inference logic + multi-person tracking)...")
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
            
            if detector is not None:
                detections = detector.detect(frame)
                frame = detector.draw_results(frame, detections)
                
                with status_lock:
                    current_status['people_detected'] = len(detections)
                    current_status['detections'] = [
                        {
                            'id': d['track_id'],
                            'status': Config.CLASS_NAMES.get(d['prediction'], 'Tracking') if d['status'] == 'classified' else 'Tracking',
                            'confidence': d['confidence'],
                            'is_fall': d.get('is_fallen', False),
                            'incident_id': d.get('incident_id')
                        }
                        for d in detections
                    ]
                    
                frame_count += 1
                if frame_count % 10 == 0:
                    current_time = time.time()
                    fps_value = 10 / (current_time - fps_time)
                    fps_time = current_time
                    current_status['fps'] = round(fps_value, 1)
                
                cv2.putText(frame, f"FPS: {fps_value:.1f}", 
                           (frame.shape[1] - 200, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
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
        'camera_available': cam_available,
        'model_type': 'EnhancedSpatial1DCNN (Inference Logic)'
    })


@app.route('/incidents', methods=['GET'])
def get_incidents():
    """Get fall incidents with optional filters"""
    try:
        db = get_database()
        
        status_filter = request.args.get('status')
        limit = int(request.args.get('limit', 100))
        
        if status_filter:
            incidents = db.get_all_incidents(limit=limit, status=status_filter)
        else:
            incidents = db.get_all_incidents(limit=limit)
        
        return jsonify({
            'success': True,
            'incidents': incidents,
            'count': len(incidents)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/incidents/statistics', methods=['GET'])
def get_incident_statistics():
    """Get incident statistics"""
    try:
        db = get_database()
        stats = db.get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/incidents/<int:incident_id>/resolve', methods=['POST'])
def resolve_incident(incident_id):
    """Manually resolve an incident"""
    try:
        db = get_database()
        db.resolve_fall_incident(incident_id)
        return jsonify({
            'success': True,
            'message': f'Incident {incident_id} resolved'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/incidents/<int:incident_id>', methods=['DELETE'])
def delete_incident(incident_id):
    """Delete a specific incident"""
    try:
        db = get_database()
        db.delete_incident(incident_id)
        return jsonify({
            'success': True,
            'message': f'Incident {incident_id} deleted'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/incidents/clear', methods=['POST'])
def clear_all_incidents():
    """Clear all incidents from database"""
    try:
        db = get_database()
        db.clear_all_incidents()
        return jsonify({
            'success': True,
            'message': 'All incidents cleared'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CAIretaker Backend - Inference Logic + Multi-Person Tracking")
    print("="*60)
    
    if initialize_detector():
        print("\nInitializing camera at startup...")
        camera_success = initialize_camera()
        
        if not camera_success:
            print("\n⚠ Warning: Camera initialization failed.")
            print("  The system will continue to attempt camera connection.")
            print("  Video feed will be available once camera is detected.\n")
        
        print("\nStarting Flask server...")
        print("Backend will be available at: http://localhost:5000")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("\n✗ Failed to initialize. Please check model files.")