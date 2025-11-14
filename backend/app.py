"""
Flask Backend for CAIretaker - Enhanced Fall Detection with Multi-Person Tracking
Exact same detection logic as inference code + person tracking with IDs and database logging

FIXES IMPLEMENTED:
1. Reset-on-Recovery: Monitoring timer resets completely when person recovers
   - Eliminates race conditions and state corruption bugs
   - Ensures consistent monitoring phase for every fall detection
   
2. Three-Tier Confidence System:
   - HIGH (≥75%): Triggers monitoring → Alert if confirmed
   - AT RISK (60-74%): Visual warning only, NO monitoring/alert
   - REJECTED (<60%): Treated as false positive
   
3. Bending Detection Override:
   - Uses leg angle analysis to distinguish bending from falling
   - Prevents false positives when bending down to pick up objects
   - Overrides high-confidence fallen predictions if bending detected
   
*** UPDATED WITH MULTI-CAMERA SWITCHING SUPPORT ***
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
    YOLO_MODEL = "yolo11n-pose.pt"
    CONFIDENCE_THRESHOLD = 0.65
    SMOOTHING_WINDOW = 7
    
    # TEMPORAL FALL DETECTION SETTINGS
    FALL_CONFIRMATION_TIME = 0.5  # Seconds person must stay fallen before alert
    FALL_CONFIRMATION_FRAMES = 3  # Minimum consecutive frames in fallen state
    
    CLASS_NAMES = {0: "Standing", 1: "Sitting", 2: "Fallen"}
    CLASS_COLORS = {
        0: (0, 255, 0),      # Standing - Green
        1: (255, 255, 0),    # Sitting - Yellow (cyan in BGR)
        2: (0, 0, 255),      # Fallen (High Confidence) - Red
        'at_risk': (0, 165, 255)  # At Risk (Low Confidence) - Orange
    }
    
    # Three-tier confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.70 # Confirmed fallen - triggers monitoring/alert (≥75%)
    LOW_CONFIDENCE_THRESHOLD = 0.50 # At risk - visual warning only (60-74%)
    # Below 0.60 = treated as normal (not fallen)
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
# SOLUTION 2: BENDING DETECTION FUNCTIONS
# ============================================================================

def calculate_leg_angles(keypoints):
    """
    Calculate average angle of legs from vertical
    
    Returns:
        float: Average leg angle in degrees (0° = vertical, 90° = horizontal)
        None if legs not visible
    
    Usage:
        - Bending: legs vertical (< 30°)
        - Fallen: legs horizontal (> 60°)
    """
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    leg_angles = []
    
    # Calculate left leg angle
    if (left_hip[2] > 0.3 and left_ankle[2] > 0.3):
        # Vector from hip to ankle (full leg)
        leg_vec = np.array([
            left_ankle[0] - left_hip[0],
            left_ankle[1] - left_hip[1]
        ])
        
        # Vertical vector (pointing down)
        vertical = np.array([0, 1])
        
        # Calculate angle from vertical
        if np.linalg.norm(leg_vec) > 1e-6:
            cos_angle = np.dot(leg_vec, vertical) / (
                np.linalg.norm(leg_vec) * np.linalg.norm(vertical)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            leg_angles.append(angle)
    
    # Calculate right leg angle
    if (right_hip[2] > 0.3 and right_ankle[2] > 0.3):
        leg_vec = np.array([
            right_ankle[0] - right_hip[0],
            right_ankle[1] - right_hip[1]
        ])
        
        vertical = np.array([0, 1])
        
        if np.linalg.norm(leg_vec) > 1e-6:
            cos_angle = np.dot(leg_vec, vertical) / (
                np.linalg.norm(leg_vec) * np.linalg.norm(vertical)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            leg_angles.append(angle)
    
    # Return average if at least one leg visible
    if len(leg_angles) > 0:
        return np.mean(leg_angles)
    return None


def calculate_torso_leg_difference(keypoints):
    """
    Calculate difference between torso angle and leg angle
    
    Returns:
        float: Angle difference in degrees
        None if cannot calculate
    
    Usage:
        - Bending: Large difference (> 40°) - torso bent, legs straight
        - Fallen: Small difference (< 20°) - both horizontal
    """
    torso_angle = calculate_body_angle(keypoints)
    leg_angle = calculate_leg_angles(keypoints)
    
    if torso_angle is not None and leg_angle is not None:
        angle_diff = abs(torso_angle - leg_angle)
        return angle_diff
    
    return None


def is_bending_posture(keypoints, image_shape):
    """
    Comprehensive check to determine if person is bending vs fallen
    
    Args:
        keypoints: (17, 3) array of keypoints
        image_shape: tuple (height, width) or (height, width, channels)
    
    Returns:
        tuple: (is_bending, confidence, reasons)
            - is_bending: True if detected as bending
            - confidence: float 0-1, confidence in the assessment
            - reasons: list of strings explaining the decision
    
    Scoring System:
        +4 points: Very strong indicator of bending
        +3 points: Strong indicator of bending
        +2 points: Moderate indicator of bending
        +1 point: Weak indicator of bending
        -2 points: Indicator of falling
        -3 points: Strong indicator of falling
        
        Total >= 4: Classified as bending
    """
    reasons = []
    bending_score = 0
    
    # Get image dimensions
    h, w = image_shape[:2] if len(image_shape) >= 2 else (1080, 1920)
    
    # 1. CHECK LEG ANGLES (CRITICAL FOR EXTREME BENDING)
    leg_angle = calculate_leg_angles(keypoints)
    if leg_angle is not None:
        if leg_angle < 35:  # Legs are relatively vertical
            bending_score += 4  # Most important indicator
            reasons.append(f"Legs vertical ({leg_angle:.0f}°)")
        elif leg_angle > 60:  # Legs are horizontal (fallen)
            bending_score -= 3
            reasons.append(f"Legs horizontal ({leg_angle:.0f}°)")
        else:  # Legs at moderate angle (35-60°)
            bending_score += 2
            reasons.append(f"Legs moderate ({leg_angle:.0f}°)")
    
    # 2. CHECK TORSO-LEG ANGLE DIFFERENCE
    angle_diff = calculate_torso_leg_difference(keypoints)
    if angle_diff is not None:
        if angle_diff > 35:  # Large difference = torso bent, legs straight
            bending_score += 3
            reasons.append(f"Torso bent, legs straight (Δ{angle_diff:.0f}°)")
        elif angle_diff < 20:  # Small difference = both horizontal
            bending_score -= 2
            reasons.append(f"Fully horizontal (Δ{angle_diff:.0f}°)")
        else:
            bending_score += 1
            reasons.append(f"Moderate bend (Δ{angle_diff:.0f}°)")
    
    # 3. CHECK ANKLE POSITIONS (CRITICAL - FEET ON GROUND = BENDING)
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    ankle_y_positions = []
    if left_ankle[2] > 0.3:
        ankle_y_positions.append(left_ankle[1] / h)
    if right_ankle[2] > 0.3:
        ankle_y_positions.append(right_ankle[1] / h)
    
    if len(ankle_y_positions) > 0:
        avg_ankle_y = np.mean(ankle_y_positions)
        if avg_ankle_y > 0.80:  # Ankles near bottom (standing/bending)
            bending_score += 3  # Feet on ground is very strong indicator
            reasons.append(f"Feet on ground ({avg_ankle_y:.2f})")
    
    # 4. CHECK HIP ELEVATION (More lenient for deep bending)
    hip_height = check_hip_position(keypoints, image_shape)
    if hip_height is not None:
        if hip_height < 0.5:  # Hips elevated
            bending_score += 2
            reasons.append(f"Hips elevated ({hip_height:.2f})")
        elif hip_height > 0.75:  # Hips very low
            bending_score -= 3
            reasons.append(f"Hips on ground ({hip_height:.2f})")
        else:  # Hips at moderate height (0.5-0.75)
            bending_score += 1
            reasons.append(f"Hips moderate ({hip_height:.2f})")
    
    # 5. CHECK KNEE POSITIONS
    knee_hip_dist, knee_height = check_knee_position(keypoints, image_shape)
    if knee_hip_dist is not None:
        if knee_hip_dist > 0.15:  # Knees bent
            bending_score += 1
            reasons.append("Knees bent")
    
    # 6. ADDITIONAL CHECK: If legs vertical AND feet on ground, DEFINITELY bending
    if leg_angle is not None and len(ankle_y_positions) > 0:
        avg_ankle_y = np.mean(ankle_y_positions)
        if leg_angle < 40 and avg_ankle_y > 0.80:
            # This combination is definitive proof of bending, not falling
            bending_score += 2  # Bonus points for this strong combination
            reasons.append("STRONG: Vertical legs + grounded feet")
    
    # CALCULATE FINAL DECISION
    is_bending = bending_score >= 4
    
    # Calculate confidence (0-1 scale)
    confidence = min(abs(bending_score) / 14.0, 1.0)
    
    return is_bending, confidence, reasons


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
        
        # Temporal fall tracking
        self.fall_candidates = defaultdict(lambda: {
            'start_time': None,
            'frame_count': 0,
            'consecutive_fallen_frames': 0
        })
        
        # Track at-risk detections for analytics
        self.at_risk_log = []
        self.rejected_log = []
        
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
                            
                            # ========================================
                            # FIX: Store RAW prediction BEFORE smoothing
                            # ========================================
                            raw_prediction = predicted.item()
                            raw_confidence = confidence_val.item()
                        
                        # Smooth predictions per track ID (for display purposes)
                        self.prediction_buffers[track_id].append(raw_prediction)
                        if len(self.prediction_buffers[track_id]) >= Config.SMOOTHING_WINDOW // 2:
                            smoothed_prediction = max(set(self.prediction_buffers[track_id]), 
                                           key=self.prediction_buffers[track_id].count)
                        else:
                            smoothed_prediction = raw_prediction
                        
                        # Use smoothed for display
                        prediction = smoothed_prediction
                        confidence = raw_confidence
                        
                        # ========================================
                        # THREE-TIER CONFIDENCE SYSTEM
                        # - High confidence (≥75%): Confirmed fallen → triggers monitoring/alert
                        # - At Risk (60-74%): Visual warning only (orange box), NO monitoring
                        # - Rejected (<60%): Treated as false positive
                        # ========================================
                        
                        # Get fallen class probability directly
                        fallen_confidence = probabilities[0][2].item()
                        
                        # Initialize variables
                        display_state = "normal"
                        confidence_tier = "N/A"
                        is_raw_fallen = False
                        
                        if raw_prediction == 2:  # Model predicts fallen class
                            if fallen_confidence >= Config.HIGH_CONFIDENCE_THRESHOLD:
                                # HIGH CONFIDENCE (≥75%): Triggers monitoring
                                is_raw_fallen = True
                                confidence_tier = "HIGH"
                                display_state = "normal"  # Will be overridden to monitoring/fallen
                                
                            elif fallen_confidence >= Config.LOW_CONFIDENCE_THRESHOLD:
                                # AT RISK (60-74%): Visual warning only
                                is_raw_fallen = False  # Does NOT trigger monitoring
                                confidence_tier = "AT_RISK"
                                display_state = "at_risk"
                                
                                print(f"⚠️ Person ID {track_id}: AT RISK (Medium confidence)")
                                print(f"   Fallen confidence: {fallen_confidence:.2%}")
                                print(f"   Visual warning only - NO monitoring/alert")
                                
                                # Log for analytics
                                self.at_risk_log.append({
                                    'timestamp': time.time(),
                                    'person_id': track_id,
                                    'confidence': fallen_confidence,
                                    'reason': 'medium_confidence_fallen'
                                })
                                
                            else:
                                # REJECTED (<60%): False positive
                                is_raw_fallen = False
                                confidence_tier = "REJECTED"
                                display_state = "normal"
                                
                                print(f"❌ Person ID {track_id}: Fallen REJECTED (low confidence)")
                                print(f"   Fallen confidence: {fallen_confidence:.2%}")
                                
                                # Log for analytics
                                self.rejected_log.append({
                                    'timestamp': time.time(),
                                    'person_id': track_id,
                                    'confidence': fallen_confidence,
                                    'reason': 'very_low_confidence'
                                })
                        else:
                            # Not predicted as fallen (class 0 or 1)
                            is_raw_fallen = False
                            confidence_tier = "N/A"
                            display_state = "normal"
                        
                        # CRITICAL FIX: Override smoothed prediction for rejected/at-risk falls
                        # This prevents display showing "Fallen" for low confidence detections
                        if raw_prediction == 2 and not is_raw_fallen:
                            # Model predicted fallen but confidence too low
                            # Override the smoothed prediction to prevent "Fallen" display
                            if confidence_tier == "AT_RISK":
                                # Keep prediction as 2 (fallen) so display can show "At Risk"
                                # But ensure is_raw_fallen stays False
                                pass  # prediction stays as smoothed value (likely 2)
                            else:
                                # REJECTED - change prediction to standing
                                prediction = 0  # Override to Standing
                        
                        # Check for fall state changes
                        was_fallen = self.fall_states[track_id]['is_fallen']
                        
                        # ========================================
                        # BENDING DETECTION OVERRIDE
                        # Only applies to HIGH confidence fallen detections
                        # ========================================
                        if is_raw_fallen:
                            # Model says fallen with high confidence, but check if it's actually bending
                            is_bending, bend_conf, reasons = is_bending_posture(keypoints, frame.shape)
                            
                            if is_bending and bend_conf > 0.5:
                                # Override prediction - it's bending, not fallen
                                print(f"✓ Person ID {track_id}: Detected BENDING (not fallen)")
                                print(f"  Confidence: {bend_conf:.2f}")
                                print(f"  Reasons: {', '.join(reasons)}")
                                
                                # Change to standing
                                prediction = 0
                                confidence = bend_conf
                                is_raw_fallen = False
                                confidence_tier = "BENDING"
                                display_state = "normal"
                        
                        # ========================================
                        # SIMPLIFIED TEMPORAL FALL DETECTION
                        # Strategy: Reset monitoring completely on any recovery
                        # Only HIGH confidence (≥75%) fallen detections trigger monitoring
                        # ========================================
                        current_time = time.time()
                        candidate = self.fall_candidates[track_id]
                        
                        if is_raw_fallen:
                            # Person detected in HIGH CONFIDENCE fallen position (≥75%)
                            
                            if was_fallen:
                                # Person is already confirmed as fallen
                                # Just maintain the state - no need to re-process
                                print(f"🔴 Person ID {track_id}: Maintaining FALLEN state")
                                
                            elif candidate['start_time'] is None:
                                # FIRST detection of high-confidence fallen position - START MONITORING
                                # This is a fresh start (either new fall or after recovery)
                                candidate['start_time'] = current_time
                                candidate['frame_count'] = 1
                                candidate['consecutive_fallen_frames'] = 1
                                
                                # CRITICAL: Force is_fallen to False at monitoring start
                                # This ensures we go through proper monitoring phase
                                if self.fall_states[track_id]['is_fallen']:
                                    print(f"⚠️ WARNING: is_fallen was True, forcing False for monitoring")
                                self.fall_states[track_id]['is_fallen'] = False
                                
                                print(f"\n{'='*60}")
                                print(f"⏱️ MONITORING STARTED - Person ID {track_id}")
                                print(f"{'='*60}")
                                print(f"Fallen confidence: {fallen_confidence:.2%} (HIGH - ≥75%)")
                                print(f"Confirmation requirements:")
                                print(f"  • Time: {Config.FALL_CONFIRMATION_TIME}s")
                                print(f"  • Frames: {Config.FALL_CONFIRMATION_FRAMES} consecutive")
                                print(f"Status: MONITORING IN PROGRESS...")
                                print(f"{'='*60}\n")
                            
                            else:
                                # Continuing in fallen position - STILL MONITORING
                                candidate['frame_count'] += 1
                                candidate['consecutive_fallen_frames'] += 1
                                elapsed_time = current_time - candidate['start_time']
                                
                                # Check if both thresholds are met
                                time_threshold_met = elapsed_time >= Config.FALL_CONFIRMATION_TIME
                                frames_threshold_met = candidate['consecutive_fallen_frames'] >= Config.FALL_CONFIRMATION_FRAMES
                                
                                if time_threshold_met and frames_threshold_met:
                                    # Both thresholds met - CONFIRM FALL
                                    if not self.fall_states[track_id]['is_fallen']:
                                        # First time confirming - TRIGGER ALERT
                                        print(f"\n{'='*70}")
                                        print(f"🚨🚨🚨 CONFIRMED FALL ALERT 🚨🚨🚨")
                                        print(f"{'='*70}")
                                        print(f"Person ID: {track_id}")
                                        print(f"Location: {Config.LOCATION_NAME}")
                                        print(f"Fallen Confidence: {fallen_confidence:.2%}")
                                        print(f"Time Fallen: {elapsed_time:.2f}s (threshold: {Config.FALL_CONFIRMATION_TIME}s ✓)")
                                        print(f"Consecutive Frames: {candidate['consecutive_fallen_frames']} (threshold: {Config.FALL_CONFIRMATION_FRAMES} ✓)")
                                        print(f"")
                                        
                                        # Log incident to database
                                        incident_id = self.db.log_fall_incident(
                                            person_id=track_id,
                                            confidence=fallen_confidence,
                                            location=Config.LOCATION_NAME
                                        )
                                        
                                        # Mark as confirmed fallen
                                        self.fall_states[track_id]['is_fallen'] = True
                                        self.fall_states[track_id]['incident_id'] = incident_id
                                        
                                        print(f"📝 Incident logged to database (ID: {incident_id})")
                                        print(f"🚨 ALERT TRIGGERED - Caregivers must respond")
                                        print(f"{'='*70}\n")
                                    else:
                                        # Already confirmed (safety net - shouldn't happen often)
                                        print(f"ℹ️ Person ID {track_id}: Fall already confirmed, maintaining state")
                                else:
                                    # Still monitoring - thresholds not yet met
                                    remaining_time = max(0, Config.FALL_CONFIRMATION_TIME - elapsed_time)
                                    remaining_frames = max(0, Config.FALL_CONFIRMATION_FRAMES - candidate['consecutive_fallen_frames'])
                                    
                                    print(f"⏱️ Person ID {track_id}: MONITORING IN PROGRESS")
                                    print(f"   Confidence: {fallen_confidence:.2%}")
                                    print(f"   Time: {elapsed_time:.2f}s / {Config.FALL_CONFIRMATION_TIME}s ({remaining_time:.2f}s remaining)")
                                    print(f"   Frames: {candidate['consecutive_fallen_frames']} / {Config.FALL_CONFIRMATION_FRAMES} ({remaining_frames} remaining)")
                        
                        else:
                            # NOT in high-confidence fallen position
                            # Could be: standing, sitting, at-risk, or low confidence fallen
                            
                            # If monitoring was active, RESET IT
                            if candidate['start_time'] is not None:
                                elapsed = current_time - candidate['start_time']
                                
                                # Person recovered during monitoring (before confirmation)
                                print(f"\n{'='*60}")
                                print(f"✓ RECOVERY DETECTED - Person ID {track_id}")
                                print(f"{'='*60}")
                                print(f"Fallen duration: {elapsed:.2f}s (threshold: {Config.FALL_CONFIRMATION_TIME}s)")
                                print(f"Frames: {candidate['consecutive_fallen_frames']} (threshold: {Config.FALL_CONFIRMATION_FRAMES})")
                                
                                if elapsed < Config.FALL_CONFIRMATION_TIME:
                                    print(f"Assessment: Brief fallen pose detected")
                                    print(f"Result: NO ALERT - Person recovered before confirmation")
                                else:
                                    print(f"Assessment: Prolonged fallen pose but insufficient frames")
                                    print(f"Result: NO ALERT - Person recovered before full confirmation")
                                
                                # COMPLETE RESET - start fresh if person falls again
                                print(f"Monitoring: RESET - Will restart from scratch on next fall")
                                print(f"{'='*60}\n")
                                
                                candidate['start_time'] = None
                                candidate['frame_count'] = 0
                                candidate['consecutive_fallen_frames'] = 0
                            
                            # Check for recovery from confirmed fall
                            if was_fallen:
                                print(f"\n{'='*70}")
                                print(f"✅✅✅ RECOVERY CONFIRMED ✅✅✅")
                                print(f"{'='*70}")
                                print(f"Person ID: {track_id}")
                                print(f"Status: Person has stood up and recovered")
                                print(f"Action: Resolving fall incident in database")
                                print(f"")
                                
                                # Resolve the incident in database
                                if self.fall_states[track_id]['incident_id'] is not None:
                                    self.db.resolve_fall_for_person(track_id)
                                    print(f"📝 Incident {self.fall_states[track_id]['incident_id']} marked as RESOLVED")
                                
                                # Clear fall state
                                self.fall_states[track_id]['is_fallen'] = False
                                self.fall_states[track_id]['incident_id'] = None
                                
                                print(f"Person ID {track_id} returned to normal monitoring")
                                print(f"{'='*70}\n")
                        # ========================================
                        
                        status = "classified"
                    else:
                        # ========================================
                        # Not enough keypoints - DO NOT classify as fallen
                        # ========================================
                        prediction = None
                        confidence = 0.0
                        status = "insufficient_keypoints"
                        
                        # Reset fall candidate if tracking was ongoing
                        if track_id in self.fall_candidates:
                            candidate = self.fall_candidates[track_id]
                            if candidate['start_time'] is not None:
                                print(f"⚠ Person ID {track_id}: Keypoints lost during monitoring, resetting")
                                candidate['start_time'] = None
                                candidate['frame_count'] = 0
                                candidate['consecutive_fallen_frames'] = 0
                        # ========================================
                    
                    detections.append({
                        'track_id': track_id,
                        'box': box,
                        'box_conf': float(box_conf),
                        'keypoints': keypoints,
                        'prediction': prediction,
                        'display_state': display_state if 'display_state' in locals() else "normal",  # New field
                        'confidence': float(confidence),
                        'confidence_tier': confidence_tier if 'confidence_tier' in locals() else "N/A",
                        'fallen_confidence': fallen_confidence if 'fallen_confidence' in locals() else 0.0,
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
            if person_id in self.fall_candidates:
                del self.fall_candidates[person_id]
        
        return detections
    
    def draw_results(self, frame, detections):
        """Draw bounding boxes, IDs, and classifications on frame with ENHANCED keypoint visibility"""
        current_time = time.time()
        
        # COCO-17 skeleton connections (body structure)
        skeleton_connections = [
            # Head to shoulders
            (0, 1), (0, 2),  # Nose to eyes
            (1, 3), (2, 4),  # Eyes to ears
            (0, 5), (0, 6),  # Nose to shoulders
            
            # Torso
            (5, 6),   # Shoulders
            (5, 11), (6, 12),  # Shoulders to hips
            (11, 12),  # Hips
            
            # Arms
            (5, 7), (7, 9),   # Left arm
            (6, 8), (8, 10),  # Right arm
            
            # Legs
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)   # Right leg
        ]
        
        for detection in detections:
            track_id = detection['track_id']
            box = detection['box']
            keypoints = detection['keypoints']
            prediction = detection['prediction']
            display_state = detection.get('display_state', 'normal')
            confidence = detection['confidence']
            confidence_tier = detection.get('confidence_tier', 'N/A')
            fallen_confidence = detection.get('fallen_confidence', 0.0)
            status = detection['status']
            is_fallen = detection['is_fallen']
            visible_count = detection['visible_count']
            
            x1, y1, x2, y2 = map(int, box)
            
            # CRITICAL FIX: Check monitoring state FIRST, before checking is_fallen
            # This ensures MONITORING always displays during the confirmation period
            candidate = self.fall_candidates.get(track_id)
            is_monitoring = candidate is not None and candidate.get('start_time') is not None
            
            # Determine color and label based on state - THREE-TIER SYSTEM
            if is_monitoring and not is_fallen:
                # Currently in MONITORING phase (high confidence fallen, not yet confirmed)
                elapsed = current_time - candidate['start_time']
                remaining = Config.FALL_CONFIRMATION_TIME - elapsed
                color = (0, 165, 255)  # ORANGE for monitoring
                label = f"ID {track_id}: MONITORING ({remaining:.1f}s)"
                box_thickness = 3
            elif is_fallen:
                # Confirmed fall (monitoring completed)
                color = (0, 0, 255)  # RED for confirmed fall
                label = f"ID {track_id}: FALLEN (ALERT)"
                box_thickness = 4
            elif display_state == 'at_risk':
                # At Risk (low confidence fallen) - visual warning only
                color = Config.CLASS_COLORS['at_risk']  # ORANGE
                label = f"ID {track_id}: At Risk ({fallen_confidence:.0%})"
                box_thickness = 2
            elif prediction is not None:
                # Normal classification (standing, sitting)
                color = Config.CLASS_COLORS.get(prediction, (255, 255, 255))
                class_name = Config.CLASS_NAMES.get(prediction, "Unknown")
                label = f"ID {track_id}: {class_name} ({confidence:.0%})"
                box_thickness = 2
            else:
                # Tracking only (insufficient keypoints)
                color = (128, 128, 128)  # GRAY for tracking only
                label = f"ID {track_id}: Tracking ({visible_count} kpts)"
                box_thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # ========================================
            # ENHANCED KEYPOINT AND SKELETON DRAWING
            # ========================================
            if prediction is not None:
                # STEP 1: Draw skeleton lines FIRST (so they appear behind keypoints)
                for start_idx, end_idx in skeleton_connections:
                    if (keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3):
                        pt1 = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                        pt2 = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                        
                        # Draw thicker lines with slight transparency effect
                        # Main line (thicker, colored)
                        cv2.line(frame, pt1, pt2, color, thickness=4, lineType=cv2.LINE_AA)
                        
                        # Inner line (thinner, brighter) for contrast
                        line_color_bright = tuple(min(c + 50, 255) for c in color)
                        cv2.line(frame, pt1, pt2, line_color_bright, thickness=2, lineType=cv2.LINE_AA)
                
                # STEP 2: Draw keypoints ON TOP of lines
                for i, kp in enumerate(keypoints):
                    x_kp, y_kp, conf = kp
                    if conf > 0.3:
                        kp_pos = (int(x_kp), int(y_kp))
                        
                        # Draw outer circle (darker border for contrast)
                        cv2.circle(frame, kp_pos, 8, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                        
                        # Draw main keypoint circle (colored)
                        cv2.circle(frame, kp_pos, 6, color, -1, lineType=cv2.LINE_AA)
                        
                        # Draw inner highlight (white dot for visibility)
                        cv2.circle(frame, kp_pos, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        
        return frame


# ============================================================================
# CAMERA & GLOBAL STATE - *** UPDATED WITH MULTI-CAMERA SUPPORT ***
# ============================================================================

# Global variables
camera = None
camera_initialized = False
detector = None
current_camera_index = 0  # NEW: Track which camera is active
current_status = {
    'people_detected': 0,
    'detections': [],
    'fps': 0
}

# Thread safety
import threading
camera_lock = threading.Lock()
status_lock = threading.Lock()


def initialize_detector():
    """Initialize the fall detector (loads models)"""
    global detector
    try:
        detector = FallDetector()
        print("✓ Detector initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing detector: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_camera(camera_index=0):
    """Initialize camera with specified index"""
    global camera, camera_initialized, current_camera_index
    
    with camera_lock:
        if camera is not None:
            camera.release()
        
        try:
            camera = cv2.VideoCapture(camera_index)
            
            if not camera.isOpened():
                print(f"✗ Failed to open camera {camera_index}")
                camera = None
                camera_initialized = False
                return False
            
            # Set camera properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test read
            success, test_frame = camera.read()
            if not success or test_frame is None:
                print(f"✗ Camera {camera_index} opened but cannot read frames")
                camera.release()
                camera = None
                camera_initialized = False
                return False
            
            current_camera_index = camera_index
            camera_initialized = True
            print(f"✓ Camera {camera_index} initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error initializing camera {camera_index}: {e}")
            camera = None
            camera_initialized = False
            return False


def generate_frames():
    """Generator function for video streaming"""
    global camera, camera_initialized
    
    if not camera_initialized:
        print("⚠ Camera not initialized, attempting to initialize...")
        if not initialize_camera(current_camera_index):
            print("✗ Camera initialization failed, sending error frame...")
            
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Unavailable", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(error_frame, "Please check camera connection", (80, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
    
    print(f"✓ Starting frame generation for camera {current_camera_index} (inference logic + multi-person tracking)...")
    frame_count = 0
    fps_time = time.time()
    fps_value = 0
    consecutive_failures = 0
    
    while True:
        try:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    print("⚠ Camera lost, attempting to reconnect...")
                    if not initialize_camera(current_camera_index):
                        time.sleep(1)
                        continue
                
                success, frame = camera.read()
            
            if not success or frame is None:
                consecutive_failures += 1
                print(f"⚠ Frame read failed (attempt {consecutive_failures})")
                
                if consecutive_failures > 10:
                    print("✗ Too many consecutive failures, reinitializing camera...")
                    camera_initialized = False
                    if not initialize_camera(current_camera_index):
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
                            'status': 'At Risk' if d.get('display_state') == 'at_risk' else Config.CLASS_NAMES.get(d['prediction'], 'Tracking') if d['status'] == 'classified' else 'Tracking',
                            'confidence': d['confidence'],
                            'confidence_tier': d.get('confidence_tier', 'N/A'),
                            'is_fall': d.get('is_fallen', False),
                            'is_at_risk': d.get('display_state') == 'at_risk',  # New field
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
                
                # UPDATED to show camera index
                cv2.putText(frame, f"FPS: {fps_value:.1f} | Camera: {current_camera_index}", 
                           (frame.shape[1] - 350, 40),
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

# ============================================================================
# API ENDPOINTS - *** UPDATED WITH CAMERA SWITCHING ***
# ============================================================================

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    print(f"📹 Video feed requested for camera {current_camera_index}")
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    """NEW ENDPOINT: Switch to a different camera"""
    global current_camera_index, camera_initialized
    
    try:
        data = request.get_json()
        new_camera_index = data.get('camera_index', 0)
        
        print(f"🔄 Switching from camera {current_camera_index} to camera {new_camera_index}")
        
        camera_initialized = False
        if initialize_camera(new_camera_index):
            return jsonify({
                'success': True,
                'message': f'Switched to camera {new_camera_index}',
                'camera_index': current_camera_index
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to switch to camera {new_camera_index}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/available_cameras', methods=['GET'])
def get_available_cameras():
    """NEW ENDPOINT: Get list of available camera indices"""
    available = []
    
    # Test cameras 0-9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return jsonify({
        'success': True,
        'cameras': available,
        'current_camera': current_camera_index
    })


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
        'current_camera_index': current_camera_index, # <-- UPDATED
        'model_type': 'EnhancedSpatial1DCNN (Reset-on-Recovery + 75% Threshold + Multi-Camera)' # <-- UPDATED
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


# ============================================================================
# *** ORIGINAL ANALYTICS ENDPOINTS (PRESERVED) ***
# ============================================================================

@app.route('/analytics/at-risk', methods=['GET'])
def get_at_risk_analytics():
    """Get analytics for at-risk and rejected detections"""
    global detector
    
    if detector is None:
        return jsonify({
            'success': False,
            'error': 'Detector not initialized'
        }), 500
    
    try:
        # Get recent logs (last 1000 entries)
        at_risk_recent = detector.at_risk_log[-1000:] if len(detector.at_risk_log) > 0 else []
        rejected_recent = detector.rejected_log[-1000:] if len(detector.rejected_log) > 0 else []
        
        # Calculate statistics
        current_time = time.time()
        last_hour = current_time - 3600
        
        at_risk_last_hour = [x for x in at_risk_recent if x['timestamp'] > last_hour]
        rejected_last_hour = [x for x in rejected_recent if x['timestamp'] > last_hour]
        
        return jsonify({
            'success': True,
            'analytics': {
                'at_risk': {
                    'total': len(detector.at_risk_log),
                    'last_hour': len(at_risk_last_hour),
                    'recent': at_risk_recent[-10:],  # Last 10 events
                    'avg_confidence': np.mean([x['confidence'] for x in at_risk_recent]) if at_risk_recent else 0
                },
                'rejected': {
                    'total': len(detector.rejected_log),
                    'last_hour': len(rejected_last_hour),
                    'recent': rejected_recent[-10:],  # Last 10 events
                    'avg_confidence': np.mean([x['confidence'] for x in rejected_recent]) if rejected_recent else 0
                },
                'summary': {
                    'total_fallen_detections': len(detector.at_risk_log) + len(detector.rejected_log),
                    'at_risk_rate': len(detector.at_risk_log) / max(1, len(detector.at_risk_log) + len(detector.rejected_log)),
                    'rejection_rate': len(detector.rejected_log) / max(1, len(detector.at_risk_log) + len(detector.rejected_log))
                }
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/analytics/clear', methods=['POST'])
def clear_analytics():
    """Clear at-risk and rejected logs"""
    global detector
    
    if detector is None:
        return jsonify({
            'success': False,
            'error': 'Detector not initialized'
        }), 500
    
    try:
        detector.at_risk_log.clear()
        detector.rejected_log.clear()
        
        return jsonify({
            'success': True,
            'message': 'Analytics logs cleared'
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
    print("CAIretaker Backend - Multi-Camera Support Enabled")
    print("="*60)
    
    if initialize_detector():
        print("\nInitializing default camera at startup...")
        camera_success = initialize_camera(0) # <-- UPDATED
        
        if not camera_success:
            print("\n⚠ Warning: Camera initialization failed.")
            print("  The system will continue to attempt camera connection.")
            print("  Video feed will be available once camera is detected.\n")
        
        print("\nStarting Flask server...")
        print("Backend will be available at: http://localhost:5000")
        # --- NEW PRINT STATEMENTS ---
        print("\nNew endpoints:")
        print("  POST /switch_camera - Switch between cameras")
        print("  GET /available_cameras - Get list of available cameras")
        # ----------------------------
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("\n✗ Failed to initialize. Please check model files.")