from flask import Flask, request, jsonify, session, render_template
import numpy as np
import cv2
import base64
from mtcnn import MTCNN
import mysql.connector
from flask_cors import CORS
import os
import random
import string
from sklearn import svm
import pickle
import math
from facenet_pytorch import InceptionResnetV1
import torch
from datetime import datetime
import logging
import sys
import uuid
import dlib
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from collections import deque
from scipy.spatial import distance
import joblib
import threading
import time
from PIL import Image
import io
from torchvision.models import efficientnet_b0

# Configure logging - console only, no file, and use ASCII only for Windows
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)  # Allow cross-origin

# CDCN disabled - using enhanced screen detection instead
cdcn_model = None
logger.info("[INFO] CDCN disabled - using enhanced screen detection for anti-spoofing")

# Global variables for liveness detection
EAR_THRESHOLD = 0.2  # Increased threshold for better blink detection
CONSECUTIVE_FRAMES_THRESHOLD = 2  # Reduced for more sensitivity
CHALLENGE_EXPIRY_SECONDS = 300  # 5 minutes
BASE_LIVENESS_THRESHOLD = 0.7
MEDIUM_VALUE_THRESHOLD = 0.75  # For transactions > RM100
HIGH_VALUE_THRESHOLD = 0.85   # For transactions > RM1000

def get_liveness_threshold(transaction_amount):
    try:
        amount = float(transaction_amount)
        if amount > 1000:
            return 0.90  # Stricter for high-value transactions
        elif amount > 100:
            return 0.85  # Medium threshold
        return 0.80  # Stricter base threshold
    except (TypeError, ValueError):
        logger.warning(f"Invalid transaction_amount: {transaction_amount}, using base threshold")
        return 0.80

# Load models
detector = MTCNN()

# Load dlib's facial landmark detector (for eye detection and blink detection)
try:
    face_landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()
    logger.info("[OK] dlib facial landmark detector loaded successfully.")
except Exception as e:
    logger.error(f"Error loading dlib shape predictor: {e}")
    logger.info("Please download the shape predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    logger.info("Extract it and place in the same directory as this script")
    exit(1)

# Load FaceNet model using facenet-pytorch
try:
    facenet_model = InceptionResnetV1(pretrained='casia-webface', classify=False, num_classes=128).eval()
    logger.info("[OK] FaceNet model loaded successfully using facenet-pytorch.")
except Exception as e:
    logger.error(f"Error loading FaceNet model: {e}")
    exit(1)

# Define Enhanced Liveness Model for Deepfake Detection
class EnhancedLivenessNet(nn.Module):
    def __init__(self):
        super(EnhancedLivenessNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 2)  # 2 outputs: real or fake
        )

    def forward(self, x):
        logits = self.backbone(x)
        return F.softmax(logits, dim=1)

# Load or initialize the liveness detection model
liveness_model = EnhancedLivenessNet()
LIVENESS_MODEL_PATH = 'models/enhanced_liveness_model.pth'

if os.path.exists(LIVENESS_MODEL_PATH):
    try:
        liveness_model.load_state_dict(torch.load(LIVENESS_MODEL_PATH, map_location=torch.device('cpu')))
        liveness_model.eval()
        logger.info("[OK] Enhanced liveness detection model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading enhanced liveness model: {e}")
        logger.info("Will use other liveness detection methods instead.")
else:
    logger.warning(f"Enhanced liveness model file {LIVENESS_MODEL_PATH} not found. Using other methods.")

# Optical flow analysis model
try:
    optical_flow_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    logger.info("[OK] Optical flow parameters configured.")
except Exception as e:
    logger.error(f"Error configuring optical flow: {e}")

# Connect to MySQL
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Change if needed
        database="payment_facial"
    )

# Store for active liveness challenges
active_challenges = {}

# Preprocessing: Standardize image for FaceNet
def preprocess_face(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    logger.debug(f"Preprocessing - Mean: {mean:.2f}, Std: {std:.2f}")
    face_pixels = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2RGB)
    face_pixels = torch.tensor(face_pixels).permute(2, 0, 1)
    face_pixels = face_pixels.float() / 255.0
    face_pixels = (face_pixels - 0.5) / 0.5
    return face_pixels

# Preprocessing for enhanced liveness detection
def preprocess_for_liveness(face_image):
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_image = transform(pil_image).unsqueeze(0)
    return tensor_image

# Convert base64 image to OpenCV
def base64_to_image(base64_str):
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is not None:
            height, width = image.shape[:2]
            logger.info(f"Input image resolution: {width}x{height}")
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

# Face alignment (based on eyes position)
def align_face(image, keypoints):
    try:
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))
        return aligned
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}, using original image")
        return image

# Extract face and align it
def extract_face(image):
    height, width = image.shape[:2]
    if height * width < 400 * 300:
        return None, f"Image resolution too low. Minimum required: 400x300 pixels (got {width}x{height})."
    detections = detector.detect_faces(image)
    if not detections:
        return None, "No face detected"
    best = max(detections, key=lambda x: x['confidence'])
    confidence = best['confidence']
    logger.info(f"Face detection confidence: {confidence:.2f}")
    if confidence < 0.85:
        return None, f"Low face detection confidence: {confidence:.2f}"
    x1, y1, w, h = best['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h
    face = image[y1:y2, x1:x2]
    if w * h < 5000:
        return None, "Detected face too small"
    face_aligned = align_face(face, best['keypoints'])
    face_resized = cv2.resize(face_aligned, (160, 160))
    return face_resized, None

# Normalize embedding to unit length
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

# Get embedding using FaceNet model
def get_embedding(face):
    face_tensor = preprocess_face(face)
    face_tensor = face_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = facenet_model(face_tensor)
    embedding = embedding.squeeze().numpy()
    embedding = np.array(embedding, dtype=np.float32)
    embedding = normalize_embedding(embedding)
    logger.debug(f"Generated embedding: {embedding[:5]}... (Norm: {np.linalg.norm(embedding):.2f})")
    return embedding

# Convert dlib rectangle to opencv rectangle format
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Convert dlib shape to numpy array of (x, y) coordinates
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Calculate Eye Aspect Ratio (EAR) for blink detection
def eye_aspect_ratio(eye):
    try:
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        if C == 0:
            return 0.0
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        logger.warning(f"Error calculating EAR: {e}")
        return 0.0

# Calculate Mouth Aspect Ratio (MAR) for smile detection
def mouth_aspect_ratio(mouth):
    try:
        horizontal = np.linalg.norm(mouth[0] - mouth[6])
        vertical = np.linalg.norm(mouth[3] - mouth[9])
        if horizontal == 0:
            return 0.0
        mar = vertical / horizontal
        return mar
    except Exception as e:
        logger.warning(f"Error calculating MAR: {e}")
        return 0.0

# Liveness detection: Blink detection
def detect_blinks(frame_sequence, threshold=EAR_THRESHOLD, consecutive_frames=CONSECUTIVE_FRAMES_THRESHOLD):
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)
    blink_counter = 0
    ear_values = []
    counter = 0
    ear_history = []
    logger.info(f"Starting blink detection on {len(frame_sequence)} frames")
    for i, frame in enumerate(frame_sequence):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rects = face_detector(gray, 0)
            if len(rects) > 0:
                rect = rects[0]
                shape = face_landmark_detector(gray, rect)
                shape = shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                ear_values.append(ear)
                ear_history.append(ear)
                logger.debug(f"Frame {i} - Left EAR: {leftEAR:.3f}, Right EAR: {rightEAR:.3f}, Avg EAR: {ear:.3f}")
                if ear < threshold:
                    counter += 1
                    logger.debug(f"Frame {i} - EAR {ear:.3f} below threshold {threshold:.3f}, counter: {counter}")
                else:
                    if counter >= consecutive_frames:
                        blink_counter += 1
                        logger.info(f"Blink detected at frame {i}, EAR: {ear:.3f}, counter: {counter}")
                    counter = 0
            else:
                logger.debug(f"Frame {i} - No face detected")
        except Exception as e:
            logger.warning(f"Error in frame {i}: {e}")
            continue
    if len(ear_values) == 0:
        logger.warning("No valid EAR values, checking for alternative blink detection")
        return 0, []
    avg_ear = sum(ear_values) / len(ear_values)
    ear_std = np.std(ear_values)
    adaptive_threshold = avg_ear * 0.8
    logger.info(f"Adaptive threshold: {adaptive_threshold:.3f} (0.8 * avg_ear {avg_ear:.3f})")
    if blink_counter == 0 and len(ear_history) > 5:
        logger.info("No blinks detected with direct method, searching for dips in EAR...")
        for i in range(2, len(ear_history) - 2):
            if (ear_history[i] < ear_history[i-1] and 
                ear_history[i] < ear_history[i-2] and
                ear_history[i] < ear_history[i+1] and
                ear_history[i] < ear_history[i+2] and
                ear_history[i] < adaptive_threshold):
                blink_counter += 1
                logger.info(f"Blink detected using pattern analysis at position {i}, EAR: {ear_history[i]:.3f}")
    logger.info(f"Blink detection summary: {blink_counter} blinks, Avg EAR: {avg_ear:.3f}, EAR StdDev: {ear_std:.3f}")
    return blink_counter, ear_values

# Smile detection function
def detect_smile(frame_sequence):
    mouth_indices = list(range(48, 68))
    mar_values = []
    smile_detected = False
    smile_confidence = 0.0
    logger.info(f"Starting smile detection on {len(frame_sequence)} frames")
    for i, frame in enumerate(frame_sequence):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = face_detector(gray, 0)
            if len(rects) > 0:
                rect = rects[0]
                shape = face_landmark_detector(gray, rect)
                shape = shape_to_np(shape)
                mouth = shape[mouth_indices]
                mar = mouth_aspect_ratio(mouth)
                mar_values.append(mar)
                logger.debug(f"Frame {i} - Mouth Aspect Ratio: {mar:.3f}")
                face_width = rect.right() - rect.left()
                mouth_width = mouth[6][0] - mouth[0][0]
                width_ratio = mouth_width / face_width
                logger.debug(f"Frame {i} - Face width: {face_width}, Mouth width: {mouth_width}, Width ratio: {width_ratio:.3f}")
                if mar < 0.5 and width_ratio > 0.4:
                    smile_confidence = max(smile_confidence, 0.7)
                    logger.debug(f"Frame {i} - Potential smile detected, confidence: 0.7")
            else:
                logger.debug(f"Frame {i} - No face detected")
        except Exception as e:
            logger.warning(f"Error in smile detection frame {i}: {e}")
            continue
    if len(mar_values) > 3:
        avg_mar = sum(mar_values) / len(mar_values)
        mar_min = min(mar_values) 
        mar_max = max(mar_values)
        mar_range = mar_max - mar_min
        logger.info(f"Smile analysis - Avg MAR: {avg_mar:.3f}, Min: {mar_min:.3f}, Max: {mar_max:.3f}, Range: {mar_range:.3f}")
        if mar_range > 0.15:
            smile_detected = True
            smile_confidence = max(smile_confidence, 0.9)
            logger.info(f"Smile detected - significant MAR range: {mar_range:.3f} > 0.15, confidence: 0.9")
        elif mar_min < 0.5 and avg_mar < 0.6:
            smile_detected = True
            smile_confidence = max(smile_confidence, 0.8)
            logger.info(f"Smile detected - sustained smile pattern, min MAR: {mar_min:.3f}, avg MAR: {avg_mar:.3f}, confidence: 0.8")
    logger.info(f"Smile detection result: {smile_detected}, Confidence: {smile_confidence:.2f}")
    return smile_detected, smile_confidence

# Facial expression/challenge verification
def verify_facial_expression(frame_sequence, expression_type):
    if expression_type == 'smile':
        smile_detected, confidence = detect_smile(frame_sequence)
        natural_movement, flow_score, _ = analyze_optical_flow(frame_sequence)
        is_valid = smile_detected and natural_movement
        score = min(confidence * 0.7 + flow_score * 0.3, 1.0)
        details = f"Smile detected: {smile_detected}, confidence: {confidence:.2f}, natural_movement: {natural_movement}"
    elif expression_type == 'blink':
        blink_count, _ = detect_blinks(frame_sequence)
        natural_movement, flow_score, _ = analyze_optical_flow(frame_sequence)
        is_valid = blink_count >= 1 and natural_movement
        score = min(min(blink_count / 2.0, 1.0) * 0.7 + flow_score * 0.3, 1.0)
        details = f"Blinks detected: {blink_count}, natural_movement: {natural_movement}"
    elif expression_type in ['head_left', 'head_right']:
        movement_detected, movement_range = detect_head_movement(frame_sequence, 'horizontal')
        natural_movement, flow_score, _ = analyze_optical_flow(frame_sequence)
        is_valid = movement_detected and natural_movement
        score = min(movement_range / 40.0 * 0.7 + flow_score * 0.3, 1.0)
        details = f"Horizontal movement detected: {movement_detected}, range: {movement_range}, natural_movement: {natural_movement}"
    elif expression_type in ['head_up', 'head_down']:
        movement_detected, movement_range = detect_head_movement(frame_sequence, 'vertical')
        natural_movement, flow_score, _ = analyze_optical_flow(frame_sequence)
        is_valid = movement_detected and natural_movement
        score = min(movement_range / 40.0 * 0.7 + flow_score * 0.3, 1.0)
        details = f"Vertical movement detected: {movement_detected}, range: {movement_range}, natural_movement: {natural_movement}"
    else:
        is_valid, score, details = verify_liveness_multimethod(frame_sequence)
    return is_valid, score, details

# Liveness detection: Head movement detection
def detect_head_movement(frame_sequence, movement_type='horizontal'):
    positions = []
    logger.info(f"Starting {movement_type} head movement detection on {len(frame_sequence)} frames")
    for i, frame in enumerate(frame_sequence):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = face_detector(gray, 0)
            if len(rects) > 0:
                shape = face_landmark_detector(gray, rects[0])
                shape = shape_to_np(shape)
                nose_tip = shape[30]
                positions.append(nose_tip)
                logger.debug(f"Frame {i} - Nose position: {nose_tip}")
            else:
                logger.debug(f"Frame {i} - No face detected")
        except Exception as e:
            logger.warning(f"Error in frame {i} during head movement detection: {e}")
    if len(positions) < 3:
        logger.warning(f"Not enough valid positions for head movement detection: {len(positions)}")
        return False, 0
    positions = np.array(positions)
    if movement_type == 'horizontal':
        x_positions = positions[:, 0]
        x_range = max(x_positions) - min(x_positions)
        x_movement = x_range > 25
        logger.info(f"Horizontal movement analysis - Min X: {min(x_positions)}, Max X: {max(x_positions)}, Range: {x_range}")
        logger.info(f"Horizontal movement detection result: {x_movement}, Range: {x_range}")
        return x_movement, x_range
    elif movement_type == 'vertical':
        y_positions = positions[:, 1]
        y_range = max(y_positions) - min(y_positions)
        y_movement = y_range > 25
        logger.info(f"Vertical movement analysis - Min Y: {min(y_positions)}, Max Y: {max(y_positions)}, Range: {y_range}")
        logger.info(f"Vertical movement detection result: {y_movement}, Range: {y_range}")
        return y_movement, y_range
    return False, 0

# Improved optical flow analysis
def analyze_optical_flow(frame_sequence):
    if len(frame_sequence) < 5:
        return False, 0.0, "Not enough frames for optical flow analysis"
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frame_sequence]
    blurred_frames = [cv2.GaussianBlur(frame, (5, 5), 0) for frame in gray_frames]
    flow_scores = []
    flow_directions = []
    for i in range(1, len(blurred_frames)):
        prev_frame = blurred_frames[i-1]
        curr_frame = blurred_frames[i]
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_flow = np.mean(mag)
        median_flow = np.median(mag)
        max_flow = np.max(mag)
        std_flow = np.std(mag)
        flow_dir_hist, _ = np.histogram(ang, bins=8, range=(0, 2*np.pi))
        flow_dir_hist = flow_dir_hist / np.sum(flow_dir_hist + 1e-10)
        dir_entropy = -np.sum(flow_dir_hist * np.log2(flow_dir_hist + 1e-10))
        flow_scores.append({
            'mean': mean_flow,
            'median': median_flow,
            'max': max_flow,
            'std': std_flow,
            'dir_entropy': dir_entropy
        })
        flow_directions.append(np.argmax(flow_dir_hist))
    if not flow_scores:
        return False, 0.0, "Failed to calculate optical flow"
    mean_flows = [score['mean'] for score in flow_scores]
    std_flows = [score['std'] for score in flow_scores]
    dir_entropies = [score['dir_entropy'] for score in flow_scores]
    avg_mean_flow = np.mean(mean_flows)
    avg_std_flow = np.mean(std_flows)
    avg_dir_entropy = np.mean(dir_entropies)
    dir_changes = sum(1 for i in range(1, len(flow_directions)) 
                     if flow_directions[i] != flow_directions[i-1])
    logger.info(f"Enhanced optical flow - Mean flow: {avg_mean_flow:.4f}, "
                f"Std: {avg_std_flow:.4f}, Dir entropy: {avg_dir_entropy:.2f}, "
                f"Dir changes: {dir_changes}")
    if avg_mean_flow < 0.1:
        return False, 0.0, f"Insufficient movement (avg flow: {avg_mean_flow:.4f})"
    if avg_dir_entropy < 0.6 or dir_changes < 3:
        return False, 0.0, f"Uniform flow direction (entropy: {avg_dir_entropy:.2f}, changes: {dir_changes})"
    is_natural = (0.1 <= avg_mean_flow <= 5.0 and 
                  avg_std_flow > 0.05 and 
                  avg_dir_entropy > 0.6 and 
                  dir_changes >= 3)
    movement_score = min(max(avg_mean_flow / 2.0, 0.0), 1.0) * 0.4 + \
                    min(avg_std_flow * 3.0, 1.0) * 0.3 + \
                    min(avg_dir_entropy / 1.5, 1.0) * 0.2 + \
                    min(dir_changes / 5.0, 1.0) * 0.1
    details = (f"Flow: mean={avg_mean_flow:.2f}, std={avg_std_flow:.2f}, "
               f"dir_entropy={avg_dir_entropy:.2f}, dir_changes={dir_changes}")
    logger.info(f"Enhanced optical flow result: {is_natural}, Score: {movement_score:.2f}, {details}")
    return is_natural, movement_score, details

# Frequency domain analysis for texture assessment
def frequency_domain_liveness(face_image):
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        
        # FIX: Use & instead of 'and' for numpy arrays
        low_freq_mask = ((y - center_row)**2 + (x - center_col)**2 <= (rows//10)**2)
        mid_freq_mask = ((rows//10)**2 < (y - center_row)**2 + (x - center_col)**2) & ((y - center_row)**2 + (x - center_col)**2 <= (rows//5)**2)
        high_freq_mask = ((y - center_row)**2 + (x - center_col)**2 > (rows//5)**2)
        
        total_energy = np.sum(magnitude_spectrum)
        low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask]) / total_energy
        mid_freq_energy = np.sum(magnitude_spectrum[mid_freq_mask]) / total_energy
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask]) / total_energy
        
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        is_real = freq_ratio > 0.25 and high_freq_energy > 0.3
        
        horizontal_band = magnitude_spectrum[center_row, :]
        vertical_band = magnitude_spectrum[:, center_col]
        h_variance = np.var(horizontal_band)
        v_variance = np.var(vertical_band)
        is_real = is_real and (h_variance < 8000 and v_variance < 8000)
        
        realism_score = min(freq_ratio * 2.5, 1.0) * 0.7 + \
                        min(high_freq_energy * 2, 1.0) * 0.3
        
        logger.info(f"Enhanced frequency analysis: ratio={freq_ratio:.3f}, "
                    f"high_freq={high_freq_energy:.3f}, "
                    f"h_var={h_variance:.1f}, v_var={v_variance:.1f}, "
                    f"real={is_real}, score={realism_score:.2f}")
        return is_real, realism_score
    except Exception as e:
        logger.error(f"Error in frequency domain analysis: {e}")
        return False, 0.0

# Liveness detection: Anti-spoofing through texture analysis
def detect_spoofing(face_image):
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        lbp_scores = []
        for radius in [1, 2, 3]:
            lbp = np.zeros_like(gray)
            neighbors = 8
            for i in range(radius, gray.shape[0] - radius):
                for j in range(radius, gray.shape[1] - radius):
                    center = gray[i, j]
                    code = 0
                    for k in range(neighbors):
                        x = i + int(radius * math.cos(2 * math.pi * k / neighbors))
                        y = j - int(radius * math.sin(2 * math.pi * k / neighbors))
                        if gray[x, y] >= center:
                            code += 1 << k
                    lbp[i, j] = code
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
            hist = hist.astype('float') / (hist.sum() + 1e-7)
            hist_std = np.std(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            lbp_scores.append((hist_std, entropy))
        avg_hist_std = np.mean([score[0] for score in lbp_scores])
        avg_entropy = np.mean([score[1] for score in lbp_scores])
        is_real = avg_hist_std > 0.006 and avg_entropy > 5.0
        texture_score = min((avg_hist_std * 80) * 0.5 + (avg_entropy / 7.0) * 0.5, 1.0)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_real = is_real and laplacian_var > 150
        texture_score = min(texture_score * 0.8 + (laplacian_var / 500) * 0.2, 1.0)
        logger.info(f"Enhanced spoofing detection: {'Real' if is_real else 'Fake'}, "
                    f"Hist StdDev: {avg_hist_std:.5f}, Entropy: {avg_entropy:.2f}, "
                    f"Laplacian Var: {laplacian_var:.1f}, Score: {texture_score:.2f}")
        return is_real, texture_score
    except Exception as e:
        logger.error(f"Error in spoofing detection: {e}")
        return False, 0.0

# Enhanced deep learning liveness detection
def deep_learning_liveness(frame_sequence):
    """
    Use your trained enhanced liveness model for deepfake detection
    """
    if not os.path.exists(LIVENESS_MODEL_PATH):
        logger.warning(f"Enhanced liveness model not found at {LIVENESS_MODEL_PATH}")
        return True, 0.5
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        liveness_model.to(device)
        liveness_model.eval()  # Ensure model is in eval mode
        
        # Sample frames for analysis (3-5 frames should be enough)
        num_samples = min(5, len(frame_sequence))
        sample_indices = [i * len(frame_sequence) // num_samples for i in range(num_samples)]
        
        confidences = []
        
        for idx in sample_indices:
            try:
                frame = frame_sequence[idx]
                
                # Extract face region for the model
                face, error = extract_face(frame)
                if face is None:
                    logger.debug(f"Could not extract face from frame {idx}: {error}")
                    continue
                
                # Preprocess for your model
                face_tensor = preprocess_for_liveness(face)
                face_tensor = face_tensor.to(device)
                
                with torch.no_grad():
                    output = liveness_model(face_tensor)
                    
                    # Check the output format of your model
                    if output.shape[1] == 2:  # Binary classification [fake, real]
                        real_prob = output[0, 1].item()  # Real class probability
                    else:
                        real_prob = output[0, 0].item()  # Single output
                    
                    # Ensure probability is in valid range
                    real_prob = max(0.0, min(1.0, real_prob))
                    confidences.append(real_prob)
                    
                    logger.debug(f"Frame {idx} - Real probability: {real_prob:.3f}")
                    
            except Exception as e:
                logger.warning(f"Error processing frame {idx} in deep learning liveness: {e}")
                continue
        
        if not confidences:
            logger.warning("No valid frames processed for deep learning liveness")
            return True, 0.5  # Default to real if no frames processed
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust threshold based on your model's performance
        # You may need to tune this threshold based on your model's training
        threshold = 0.5  # Start with 0.5, adjust based on your model's performance
        is_real = avg_confidence >= threshold
        
        logger.info(f"Deep learning liveness detection: {'Real' if is_real else 'Fake'}, "
                    f"Avg confidence: {avg_confidence:.3f}, Threshold: {threshold}, "
                    f"Processed frames: {len(confidences)}")
        
        return is_real, avg_confidence
        
    except Exception as e:
        logger.error(f"Error in deep learning liveness check: {e}")
        return True, 0.5  # Default to real if error
    
def load_liveness_model():
    """Load the trained liveness model with better error handling"""
    global liveness_model
    
    try:
        liveness_model = EnhancedLivenessNet()
        
        if os.path.exists(LIVENESS_MODEL_PATH):
            # Load with CPU first, then move to device
            state_dict = torch.load(LIVENESS_MODEL_PATH, map_location='cpu')
            liveness_model.load_state_dict(state_dict)
            liveness_model.eval()
            logger.info("[OK] Enhanced liveness detection model loaded successfully.")
            
            # Test the model with a dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                test_output = liveness_model(dummy_input)
                logger.info(f"Model test successful. Output shape: {test_output.shape}")
                
        else:
            logger.warning(f"Enhanced liveness model file {LIVENESS_MODEL_PATH} not found.")
            
    except Exception as e:
        logger.error(f"Error loading enhanced liveness model: {e}")
        logger.info("Will use other liveness detection methods instead.")    

# Deepfake detection using frequency domain analysis
def detect_deepfake(face_image):
    """
    Use your trained model for deepfake detection instead of frequency analysis
    """
    try:
        # Use your trained model for deepfake detection
        if not os.path.exists(LIVENESS_MODEL_PATH):
            logger.warning(f"Trained deepfake model not found at {LIVENESS_MODEL_PATH}")
            return True, 0.7  # Default to real if model missing
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        liveness_model.to(device)
        liveness_model.eval()
        
        # Preprocess the face image for your model
        face_tensor = preprocess_for_liveness(face_image)
        face_tensor = face_tensor.to(device)
        
        with torch.no_grad():
            output = liveness_model(face_tensor)
            
            # Interpret your model's output
            # Adjust this based on your model's output format
            if output.shape[1] == 2:  # [fake_prob, real_prob]
                real_prob = output[0, 1].item()
            else:  # Single output
                real_prob = output[0, 0].item()
            
            # Ensure valid probability range
            real_prob = max(0.0, min(1.0, real_prob))
            
            # Threshold for your model (you may need to tune this)
            threshold = 0.5
            is_real = real_prob >= threshold
            
        logger.info(f"Trained model deepfake detection: {'Real' if is_real else 'Fake'}, "
                    f"confidence={real_prob:.3f}, threshold={threshold}")
        
        return is_real, real_prob
        
    except Exception as e:
        logger.error(f"Error in trained model deepfake detection: {e}")
        return True, 0.7  # Default to real if error

# Modified verify_liveness_multimethod to have adaptive thresholds
# Fix for the verify_liveness_multimethod function
# Replace the existing function with this corrected version

def verify_liveness_multimethod(frame_sequence, transaction_amount=0):
    logger.info(f"Starting liveness verification with {len(frame_sequence)} frames, transaction_amount={transaction_amount}")
    for i, frame in enumerate(frame_sequence):
        height, width = frame.shape[:2]
        logger.debug(f"Frame {i}: {width}x{height}, mean_pixel={np.mean(frame):.1f}")
    
    if len(frame_sequence) < 4:
        return False, 0.0, "Not enough frames for liveness detection"
    
    if detect_static_image(frame_sequence):
        return False, 0.0, "Static image or video replay detected"
    
    middle_frame = frame_sequence[len(frame_sequence) // 2]
    analyze_image_properties(middle_frame, "CURRENT_FRAME")
    
    is_screen, screen_details = detect_screen_simple_no_edges(middle_frame)
    logger.info(f"Screen detection result: {is_screen}, Details: {screen_details}")
    if is_screen:
        return False, 0.1, f"Video replay detected: {screen_details}"
    
    is_phone, phone_details = detect_phone_bezel(middle_frame)
    if is_phone:
        return False, 0.1, f"Phone replay detected: {phone_details}"
    
    is_replay, replay_score, replay_details = enhanced_temporal_analysis_simple(frame_sequence)
    if is_replay:
        return False, replay_score, f"Video replay detected: {replay_details}"
    
    methods_results = {}
    methods_scores = {}
    
    # Keep all original liveness detection methods
    blink_count, _ = detect_blinks(frame_sequence)
    methods_results['blink'] = blink_count >= 1
    methods_scores['blink'] = min(blink_count / 2.0, 1.0)
    
    head_move, move_range = detect_head_movement(frame_sequence)
    methods_results['head_movement'] = head_move
    methods_scores['head_movement'] = min(move_range / 40.0, 1.0)
    
    natural_movement, flow_score, flow_details = analyze_optical_flow(frame_sequence)
    methods_results['optical_flow'] = natural_movement
    methods_scores['optical_flow'] = flow_score
    
    if not natural_movement:
        return False, flow_score, f"No natural movement detected: {flow_details}"
    
    real_texture, texture_score = detect_spoofing(middle_frame)
    methods_results['texture'] = real_texture
    methods_scores['texture'] = texture_score
    
    # Keep frequency domain analysis (now fixed)
    real_freq, freq_score = frequency_domain_liveness(middle_frame)
    methods_results['frequency'] = real_freq
    methods_scores['frequency'] = freq_score
    
    # Keep deep learning liveness (for additional validation)
    if os.path.exists(LIVENESS_MODEL_PATH):
        dl_real, dl_score = deep_learning_liveness(frame_sequence)
        methods_results['deep_learning'] = dl_real
        methods_scores['deep_learning'] = dl_score
    
    # ONLY CHANGE: Use your trained model for deepfake detection
    not_deepfake, deepfake_score = detect_deepfake(middle_frame)  # Now uses your trained model
    methods_results['deepfake'] = not_deepfake
    methods_scores['deepfake'] = deepfake_score
    
    logger.info("Enhanced liveness detection results:")
    for method, result in methods_results.items():
        logger.info(f"{method}: {result}, score: {methods_scores.get(method, 0):.2f}")
    
    # Keep original scoring weights
    weights = {
        'blink': 0.15,         
        'head_movement': 0.15,  
        'optical_flow': 0.40,  
        'texture': 0.30,      
        'frequency': 0.15,
        'deep_learning': 0.15,
        'deepfake': 0.20
    }
    
    total_weight = 0
    final_score = 0
    for method, weight in weights.items():
        if method in methods_scores:
            final_score += methods_scores[method] * weight
            total_weight += weight
    
    if total_weight > 0:
        final_score = final_score / total_weight
    
    threshold = get_liveness_threshold(transaction_amount)
    threshold = max(threshold, 0.8)
    is_live = final_score >= threshold
    
    # Calculate pass ratio
    passed_tests = sum(1 for result in methods_results.values() if result)
    pass_ratio = passed_tests / len(methods_results)
    
    # Keep critical checks but with your trained model
    if not (methods_results.get('optical_flow', False) and 
            methods_results.get('texture', False) and 
            methods_results.get('deepfake', False)):  # deepfake now uses your model
        is_live = False
        details = "Failed critical anti-spoofing checks (optical flow, texture, or deepfake)"
    else:
        min_pass_ratio = 0.8 if transaction_amount > 100 else 0.7
        if pass_ratio < min_pass_ratio:
            is_live = False
            details = f"Failed {len(methods_results) - passed_tests} out of {len(methods_results)} liveness tests (pass ratio: {pass_ratio:.2f}, required: {min_pass_ratio:.2f})"
        else:
            details = ", ".join([f"{method}: {result}" for method, result in methods_results.items()])
    
    logger.info(f"Enhanced liveness assessment: {is_live}, score: {final_score:.2f}, "
                f"pass ratio: {pass_ratio:.2f}, threshold: {threshold:.2f}, "
                f"transaction_amount: {transaction_amount}")
    
    return is_live, final_score, details

# Modified verify_liveness function to pass transaction_amount
def verify_liveness(frame_sequence, transaction_amount=0):
    is_live, score, details = verify_liveness_multimethod(frame_sequence, transaction_amount)
    return is_live, details

# Improved function to detect static images or video replays
def detect_static_image(frames):
    if len(frames) < 2:
        return True
    frame_diffs = []
    identical_frames = 0
    for i in range(1, len(frames)):
        prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_frame, curr_frame)
        non_zero_count = np.count_nonzero(diff)
        avg_diff = non_zero_count / (prev_frame.shape[0] * prev_frame.shape[1])
        frame_diffs.append(avg_diff)
        if avg_diff < 0.003:
            identical_frames += 1
    if identical_frames > (len(frames) - 1) * 0.6:
        logger.warning(f"Too many identical consecutive frames: {identical_frames}/{len(frames)-1}")
        return True
    if len(frame_diffs) > 0:
        avg_diff = sum(frame_diffs) / len(frame_diffs)
        max_diff = max(frame_diffs)
        min_diff = min(frame_diffs)
        logger.info(f"Frame differences - Avg: {avg_diff:.5f}, Min: {min_diff:.5f}, Max: {max_diff:.5f}")
        if avg_diff < 0.0008:
            logger.warning(f"Average difference too low: {avg_diff:.5f}")
            return True
        std_diff = np.std(frame_diffs)
        if std_diff < 0.0004 and avg_diff < 0.008:
            logger.warning(f"Too uniform frame differences (possible video loop): StdDev: {std_diff:.5f}")
            return True
    return False

def detect_screen_physical_properties(face_image):
    try:
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        horizontal_band = magnitude[center_row, :]
        vertical_band = magnitude[:, center_col]
        h_variance = np.var(horizontal_band)
        v_variance = np.var(vertical_band)
        r_mean = np.mean(face_image[:, :, 2])
        g_mean = np.mean(face_image[:, :, 1]) 
        b_mean = np.mean(face_image[:, :, 0])
        color_temp = (r_mean + g_mean) / (b_mean + 1e-10)
        brightness_uniformity = np.std(gray) / np.mean(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_intensity = np.mean(edges)
        logger.info(f"Physical screen metrics:")
        logger.info(f"H/V variance: {h_variance:.1f}/{v_variance:.1f}")
        logger.info(f"Color temp: {color_temp:.2f}")
        logger.info(f"Brightness uniformity: {brightness_uniformity:.3f}")
        logger.info(f"Edge intensity: {edge_intensity:.2f}")
        screen_indicators = 0
        reasons = []
        if h_variance > 10000 or v_variance > 10000:
            screen_indicators += 1
            reasons.append(f"refresh_artifacts({h_variance:.0f},{v_variance:.0f})")
        if color_temp < 1.5 or color_temp > 3.0:
            screen_indicators += 1
            reasons.append(f"unnatural_color_temp({color_temp:.2f})")
        if brightness_uniformity < 0.3:
            screen_indicators += 1
            reasons.append(f"too_uniform({brightness_uniformity:.3f})")
        if edge_intensity > 15:
            screen_indicators += 1
            reasons.append(f"sharp_edges({edge_intensity:.1f})")
        if screen_indicators >= 2:
            logger.warning(f"Physical screen detected: {screen_indicators}/4 indicators - {', '.join(reasons)}")
            return True, f"Physical screen detected ({screen_indicators}/4): {', '.join(reasons)}"
        return False, f"No physical screen detected ({screen_indicators}/4 indicators)"
    except Exception as e:
        logger.error(f"Error in physical screen detection: {e}")
        return False, "Error in detection"

def detect_screen_simple_no_edges(face_image):
    try:
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        hist_r = cv2.calcHist([face_image], [2], None, [256], [0, 256])
        hist_g = cv2.calcHist([face_image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([face_image], [0], None, [256], [0, 256])
        hist_uniformity = (np.std(hist_r) + np.std(hist_g) + np.std(hist_b)) / 3
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        blue_mean = np.mean(face_image[:, :, 0])
        red_mean = np.mean(face_image[:, :, 2])
        blue_red_ratio = blue_mean / (red_mean + 1e-10)
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation)
        logger.debug(f"Enhanced screen detection metrics:")
        logger.debug(f"Laplacian variance: {laplacian_var:.1f}")
        logger.debug(f"Histogram uniformity: {hist_uniformity:.1f}")
        logger.debug(f"Edge density: {edge_density:.4f}")
        logger.debug(f"Brightness: {brightness:.1f}±{brightness_std:.1f}")
        logger.debug(f"Blue/Red ratio: {blue_red_ratio:.3f}")
        logger.debug(f"Saturation std: {sat_std:.1f}")
        screen_indicators = 0
        reasons = []
        if laplacian_var < 80:
            screen_indicators += 1
            reasons.append(f"low_texture({laplacian_var:.1f})")
        if hist_uniformity > 1400:
            screen_indicators += 1
            reasons.append(f"uniform_histogram({hist_uniformity:.1f})")
        if edge_density < 0.025:
            screen_indicators += 1
            reasons.append(f"low_edges({edge_density:.4f})")
        if brightness > 190 and brightness_std < 18:
            screen_indicators += 1
            reasons.append(f"screen_brightness({brightness:.1f},{brightness_std:.1f})")
        if blue_red_ratio > 1.3:
            screen_indicators += 1
            reasons.append(f"blue_dominance({blue_red_ratio:.3f})")
        if sat_std < 65:
            screen_indicators += 1
            reasons.append(f"low_saturation_std({sat_std:.1f})")
        if screen_indicators >= 3:
            logger.warning(f"Screen detected: {screen_indicators}/6 indicators - {', '.join(reasons)}")
            return True, f"Screen detected ({screen_indicators}/6): {', '.join(reasons)}"
        return False, f"No screen detected ({screen_indicators}/6 indicators)"
    except Exception as e:
        logger.error(f"Error in screen detection: {e}")
        return False, "Error in detection"

def detect_phone_bezel(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                contour_area = cv2.contourArea(contour)
                image_area = gray.shape[0] * gray.shape[1]
                area_ratio = contour_area / image_area
                if 0.3 < area_ratio < 0.9:
                    logger.warning(f"Rectangular screen detected, area ratio: {area_ratio:.2f}")
                    return True, f"Phone screen detected (area ratio: {area_ratio:.2f})"
        return False, "No phone bezel detected"
    except Exception as e:
        logger.error(f"Error in bezel detection: {e}")
        return False, "Error in detection"

def detect_screen_simple_focused(face_image):
    try:
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation)
        sat_mean = np.mean(saturation)
        color_variance = np.var(face_image)
        logger.debug(f"Focused detection - Sat mean: {sat_mean:.1f}, std: {sat_std:.1f}, Variance: {color_variance:.1f}")
        extreme_indicators = 0
        if sat_std < 60:
            extreme_indicators += 1
            logger.debug(f"Extreme indicator: very low sat std {sat_std:.1f}")
        if color_variance > 4600:
            extreme_indicators += 1
            logger.debug(f"Extreme indicator: extremely high variance {color_variance:.1f}")
        if sat_mean < 110 and color_variance > 4400 and sat_std < 70:
            extreme_indicators += 1
            logger.debug(f"Extreme indicator: combined extreme case")
        if extreme_indicators >= 2:
            logger.warning(f"Screen detected with extreme indicators: {extreme_indicators}")
            return True, f"Screen detected (extreme case): {extreme_indicators} indicators"
        logger.info("No extreme screen patterns detected")
        return False, "No extreme screen patterns detected"
    except Exception as e:
        logger.error(f"Error in focused screen detection: {e}")
        return False, "Error in detection"

def analyze_image_properties(face_image, label=""):
    try:
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        blue_mean = np.mean(face_image[:, :, 0])
        green_mean = np.mean(face_image[:, :, 1])
        red_mean = np.mean(face_image[:, :, 2])
        blue_ratio = blue_mean / (red_mean + green_mean + blue_mean + 1e-10)
        b_mean = np.mean(lab[:, :, 2])
        color_temp_ratio = (red_mean + green_mean) / (blue_mean + 1e-10)
        color_variance = np.var(face_image)
        logger.info(f"=== IMAGE ANALYSIS {label} ===")
        logger.info(f"Saturation: mean={sat_mean:.1f}, std={sat_std:.1f}")
        logger.info(f"RGB: R={red_mean:.1f}, G={green_mean:.1f}, B={blue_mean:.1f}")
        logger.info(f"Blue ratio: {blue_ratio:.3f}")
        logger.info(f"LAB b-channel: {b_mean:.1f}")
        logger.info(f"Color temp ratio: {color_temp_ratio:.2f}")
        logger.info(f"Color variance: {color_variance:.1f}")
        logger.info(f"=== END ANALYSIS ===")
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")

def detect_moiré_patterns_simple(face_image):
    return False, "Moiré detection disabled"

def enhanced_temporal_analysis_simple(frame_sequence):
    if len(frame_sequence) < 5:
        return False, 0.0, "Not enough frames"
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frame_sequence]
    compression_scores = []
    for gray_frame in gray_frames:
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        compression_scores.append(laplacian_var)
    if len(compression_scores) > 0:
        comp_mean = np.mean(compression_scores)
        comp_std = np.std(compression_scores)
        logger.debug(f"Temporal analysis - Compression mean: {comp_mean:.1f}, std: {comp_std:.1f}")
        if comp_mean > 3000 and comp_std < 30:
            logger.warning(f"Very obvious video compression detected")
            return True, 0.7, f"Strong video compression: mean={comp_mean:.1f}"
    frame_diffs = []
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
        avg_diff = np.mean(diff)
        frame_diffs.append(avg_diff)
    if len(frame_diffs) > 0:
        diff_std = np.std(frame_diffs)
        diff_mean = np.mean(frame_diffs)
        logger.debug(f"Temporal analysis - Frame diff mean: {diff_mean:.2f}, std: {diff_std:.2f}")
        if diff_std < 1.0 and diff_mean < 3.0:
            logger.warning(f"Very uniform frame differences - possible loop")
            return True, 0.8, f"Video loop pattern: std={diff_std:.2f}"
    return False, 0.0, "Natural temporal patterns"

# Cosine Similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Euclidean Distance
def euclidean_distance(e1, e2):
    return np.linalg.norm(e1 - e2)

# Create DB Tables
def setup_database():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        cursor.execute("SHOW COLUMNS FROM security_log LIKE 'liveness_score'")
        if cursor.fetchone() is None:
            cursor.execute("ALTER TABLE security_log ADD COLUMN liveness_score FLOAT")
            logger.info("Added missing liveness_score column to security_log table")
    except Exception as e:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255),
                action VARCHAR(50),
                status VARCHAR(20),
                similarity_score FLOAT,
                client_ip VARCHAR(50),
                liveness_score FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created security_log table with liveness_score column")
    db.commit()
    cursor.close()
    db.close()
    logger.info("[OK] Database setup done.")

# Log security events with liveness information
def log_security_event(user_id, action, status, similarity_score=None, client_ip=None, liveness_score=None):
    try:
        db = connect_db()
        cursor = db.cursor()
        try:
            cursor.execute(
                "INSERT INTO security_log (user_id, action, status, similarity_score, client_ip, liveness_score) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, action, status, similarity_score, client_ip, liveness_score)
            )
        except Exception as e:
            logger.warning(f"Error logging with liveness_score: {e}")
            cursor.execute(
                "INSERT INTO security_log (user_id, action, status, similarity_score, client_ip) VALUES (%s, %s, %s, %s, %s)",
                (user_id, action, status, similarity_score, client_ip)
            )
        db.commit()
        cursor.close()
        db.close()
        logger.info(f"Security log: {action} by {user_id if user_id else 'unknown'} - {status}")
    except Exception as e:
        logger.error(f"Logging error: {e}")

def log_frames_info(frames, prefix=""):
    if not frames:
        logger.warning(f"{prefix}No frames received")
        return
    logger.info(f"{prefix}Received {len(frames)} frames")
    for i, frame in enumerate(frames):
        if frame is not None:
            height, width = frame.shape[:2]
            logger.info(f"{prefix}Frame {i}: {width}x{height}")
        else:
            logger.warning(f"{prefix}Frame {i}: Invalid/None")

def store_multiple_embeddings(user_id, face, count=15):
    db = connect_db()
    cursor = db.cursor()
    original_embedding = get_embedding(face)
    logger.info(f"Storing original embedding for user {user_id}")
    cursor.execute(
        "INSERT INTO face_embeddings (user_id, embedding) VALUES (%s, %s)",
        (user_id, original_embedding.tobytes())
    )
    for i in range(count - 1):
        augmented = augment_face(face)
        embedding = get_embedding(augmented)
        logger.debug(f"Storing augmented embedding {i+1}/{count-1} for user {user_id}")
        cursor.execute(
            "INSERT INTO face_embeddings (user_id, embedding) VALUES (%s, %s)",
            (user_id, embedding.tobytes())
        )
    db.commit()
    cursor.close()
    db.close()

def augment_face(face):
    face_copy = face.copy()
    aug_type = random.choice(['rotation', 'brightness', 'noise', 'flip', 'blur'])
    if aug_type == 'rotation':
        angle = random.uniform(-15, 15)
        center = (80, 80)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        face_copy = cv2.warpAffine(face_copy, rot_mat, (160, 160))
    elif aug_type == 'brightness':
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-15, 15)
        face_copy = cv2.convertScaleAbs(face_copy, alpha=alpha, beta=beta)
    elif aug_type == 'noise':
        noise = np.random.normal(0, 5, face_copy.shape).astype(np.uint8)
        face_copy = cv2.add(face_copy, noise)
    elif aug_type == 'flip':
        face_copy = cv2.flip(face_copy, 1)
    elif aug_type == 'blur':
        blur_amount = random.choice([0, 1, 2])
        if blur_amount > 0:
            face_copy = cv2.GaussianBlur(face_copy, (2*blur_amount + 1, 2*blur_amount + 1), 0)
    return face_copy

def train_svm_classifier():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT user_id, embedding FROM face_embeddings")
    results = cursor.fetchall()
    if len(results) < 2:
        cursor.close()
        db.close()
        return None
    X, y = [], []
    for user_id, emb_blob in results:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        X.append(emb)
        y.append(user_id)
    logger.info(f"Training SVM with {len(X)} samples from {len(set(y))} users")
    clf = svm.SVC(kernel='rbf', probability=True, C=10.0, gamma='scale', class_weight='balanced')
    clf.fit(X, y)
    with open('svm_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    cursor.close()
    db.close()
    logger.info("[OK] SVM classifier trained.")
    return clf

def hybrid_recognition(input_embedding, user_claim=None):
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM face_embeddings")
    user_count = cursor.fetchone()[0]
    force_svm = user_count > 5 and not user_claim
    
    if (user_count <= 5 or user_claim) and not force_svm:
        logger.info(f"Using direct comparison mode for {'user claim: '+user_claim if user_claim else 'small user base'}")
        if user_claim:
            cursor.execute("SELECT embedding FROM face_embeddings WHERE user_id = %s", (user_claim,))
            cosine_threshold = 0.55
            required_matches = 3
        else:
            cursor.execute("SELECT user_id, embedding FROM face_embeddings")
            cosine_threshold = 0.55
            required_matches = 5
        embeddings = cursor.fetchall()
        cursor.close()
        db.close()
        
        if not embeddings:
            logger.warning(f"No embeddings found for {'user: '+user_claim if user_claim else 'any user'}")
            return None, 0, 0
        
        users = {}
        for rec in embeddings:
            # FIX: Use different variable name to avoid conflict
            current_user_id = user_claim if user_claim else rec[0]
            emb = np.frombuffer(rec[-1], dtype=np.float32)
            if current_user_id not in users:
                users[current_user_id] = []
            users[current_user_id].append(emb)
        
        best_user = None
        best_matches = 0
        best_avg_similarity = 0
        
        for current_user_id, embs in users.items():
            matches = 0
            total_similarity = 0
            for emb in embs:
                similarity = cosine_similarity(input_embedding, emb)
                total_similarity += similarity
                if similarity >= cosine_threshold:
                    matches += 1
                logger.debug(f"User: {current_user_id}, Similarity: {similarity:.4f}, Match: {similarity >= cosine_threshold}")
            
            avg_similarity = total_similarity / len(embs)
            logger.info(f"User: {current_user_id}, Matches: {matches}/{len(embs)}, Avg Similarity: {avg_similarity:.4f}")
            
            if matches > best_matches or (matches == best_matches and avg_similarity > best_avg_similarity):
                best_user = current_user_id
                best_matches = matches
                best_avg_similarity = avg_similarity
        
        if best_matches >= required_matches:
            return best_user, best_avg_similarity, best_matches
        else:
            return None, best_avg_similarity, best_matches
    else:
        logger.info("Using SVM classification mode")
        cursor.close()
        db.close()
        
        if not os.path.exists('svm_classifier.pkl'):
            clf = train_svm_classifier()
            if clf is None:
                logger.error("Failed to train SVM classifier")
                return None, 0, False
        else:
            try:
                with open('svm_classifier.pkl', 'rb') as f:
                    clf = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load SVM classifier: {e}")
                clf = train_svm_classifier()
                if clf is None:
                    return None, 0, False
        
        prediction = clf.predict([input_embedding])[0]
        probabilities = clf.predict_proba([input_embedding])[0]
        max_idx = np.argmax(probabilities)
        prob = probabilities[max_idx]
        
        # FIX: Use user_claim instead of user_id
        if user_claim is not None and prediction != user_claim:
            logger.warning(f"Identity mismatch: claimed {user_claim}, predicted {prediction}")
            return None, prob, 0
        
        logger.info(f"SVM Prediction: {prediction}, Probability: {prob:.2f}")
        
        estimated_matches = int(prob * 15)
        
        if prob >= 0.8:
            return prediction, prob, True
        return None, prob, False

# --------------- API Endpoints ------------------

# Generate a random challenge for liveness verification
@app.route('/get_random_challenge', methods=['POST', 'GET'])
def get_random_challenge():
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Define all possible challenges
        all_challenges = [
            {"type": "blink", "instruction": "Please blink naturally 2-3 times"},
            {"type": "smile", "instruction": "Please smile naturally"},
            {"type": "head_left", "instruction": "Please slowly turn your head to the left, then back to center"},
            {"type": "head_right", "instruction": "Please slowly turn your head to the right, then back to center"},
            {"type": "head_up", "instruction": "Please slowly look up, then back to center"},
            {"type": "head_down", "instruction": "Please slowly look down, then back to center"}
        ]
        
        # Generate a random length sequence between 2-3 challenges
        sequence_length = random.randint(2, 3)
        challenge_sequence = random.sample(all_challenges, sequence_length)
        
        # Store the challenge sequence
        active_challenges[session_id] = {
            "sequence": challenge_sequence,
            "current_index": 0,
            "created_at": datetime.now(),
            "completed": False,
            "frames": [],
            "challenge_results": []
        }
        
        # Return the first challenge and session ID
        first_challenge = challenge_sequence[0]
        
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "challenge_type": first_challenge["type"],
            "instruction": first_challenge["instruction"],
            "total_challenges": sequence_length,
            "current_challenge": 1,
            "expires_in": CHALLENGE_EXPIRY_SECONDS
        })
    
    except Exception as e:
        logger.error(f"Error generating random challenge sequence: {e}")
        return jsonify({
            "status": "failed",
            "message": f"Failed to generate challenge sequence: {str(e)}"
        })

@app.route('/next_challenge', methods=['POST'])
def next_challenge():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_challenges:
            return jsonify({"status": "failed", "message": "Invalid or expired session"})
        
        session = active_challenges[session_id]
        current_index = session["current_index"]
        
        # Verify the current challenge was completed successfully
        if len(session["challenge_results"]) <= current_index:
            return jsonify({"status": "failed", "message": "Current challenge not yet verified"})
        
        if not session["challenge_results"][current_index]:
            return jsonify({"status": "failed", "message": "Current challenge verification failed"})
        
        # Move to the next challenge
        current_index += 1
        if current_index >= len(session["sequence"]):
            # All challenges completed
            session["completed"] = True
            
            # Calculate final liveness score as average of all challenge scores
            liveness_scores = session.get("liveness_scores", [])
            final_score = sum(liveness_scores) / len(liveness_scores) if liveness_scores else 0
            session["final_liveness_score"] = final_score
            
            return jsonify({
                "status": "success",
                "message": "All challenges completed successfully",
                "completed": True,
                "liveness_verified": True,
                "liveness_score": final_score
            })
        
        # Update current challenge index
        session["current_index"] = current_index
        next_challenge = session["sequence"][current_index]
        
        # Clear frames from previous challenge to save memory
        session["frames"] = []
        
        return jsonify({
            "status": "success",
            "challenge_type": next_challenge["type"],
            "instruction": next_challenge["instruction"],
            "current_challenge": current_index + 1,
            "total_challenges": len(session["sequence"])
        })
        
    except Exception as e:
        logger.error(f"Error getting next challenge: {e}")
        return jsonify({"status": "failed", "message": f"Failed to get next challenge: {str(e)}"})

# Start a liveness challenge session
@app.route('/liveness_challenge', methods=['POST'])
def liveness_challenge():
    try:
        # Generate a unique session ID for this challenge
        session_id = str(uuid.uuid4())
        
        # Optional parameters
        data = request.json or {}
        challenge_type = data.get('challenge_type', 'blink')
        
        # Store challenge in active challenges
        active_challenges[session_id] = {
            'type': challenge_type,
            'created_at': datetime.now(),
            'completed': False,
            'frames': []
        }
        
        # Generate instruction based on challenge type
        if challenge_type == 'blink':
            instruction = "Please blink naturally 2-3 times while looking at the camera"
        elif challenge_type == 'smile':
            instruction = "Please smile naturally at the camera"
        elif challenge_type == 'head_movement':
            instruction = "Please slowly turn your head from left to right, then back to center"
        elif challenge_type == 'head_left':
            instruction = "Please slowly turn your head to the left, then back to center"
        elif challenge_type == 'head_right':
            instruction = "Please slowly turn your head to the right, then back to center"
        elif challenge_type == 'head_up':
            instruction = "Please slowly look up, then back to center"
        elif challenge_type == 'head_down':
            instruction = "Please slowly look down, then back to center"
        else:
            instruction = "Please look directly at the camera and stay still"
        
        # Return challenge instructions and session ID
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "challenge_type": challenge_type,
            "instruction": instruction,
            "expires_in": CHALLENGE_EXPIRY_SECONDS
        })
    except Exception as e:
        logger.error(f"Error creating liveness challenge: {str(e)}")
        return jsonify({"status": "failed", "message": f"Failed to create liveness challenge: {str(e)}"})

# Submit a frame for liveness challenge
@app.route('/submit_liveness_frame', methods=['POST'])
def submit_liveness_frame():
    try:
        data = request.json
        session_id = data.get('session_id')
        frame_base64 = data.get('frame')
        
        if not session_id or not frame_base64:
            return jsonify({"status": "failed", "message": "Missing session_id or frame"})
        
        if session_id not in active_challenges:
            return jsonify({"status": "failed", "message": "Invalid or expired session"})
        
        # Check if challenge is already completed
        if active_challenges[session_id]['completed']:
            return jsonify({"status": "failed", "message": "Challenge already completed"})
        
        # Convert base64 to image
        frame = base64_to_image(frame_base64)
        if frame is None:
            return jsonify({"status": "failed", "message": "Invalid frame data"})
        
        # Store frame
        active_challenges[session_id]['frames'].append(frame)
        
        return jsonify({
            "status": "success",
            "frames_received": len(active_challenges[session_id]['frames']),
            "message": "Frame received successfully"
        })
    except Exception as e:
        logger.error(f"Error submitting liveness frame: {str(e)}")
        return jsonify({"status": "failed", "message": f"Failed to submit frame: {str(e)}"})

# Modified /verify_liveness endpoint
@app.route('/verify_liveness', methods=['POST'])
def verify_liveness_challenge():
    try:
        data = request.json
        session_id = data.get('session_id')
        transaction_amount = data.get('transaction_amount', 0)
        
        logger.info(f"Verifying liveness for session {session_id}, transaction amount: {transaction_amount}")
        
        if not session_id:
            logger.warning("Missing session_id in request")
            return jsonify({"status": "failed", "message": "Missing session_id"})
        
        if session_id not in active_challenges:
            logger.warning(f"Invalid or expired session: {session_id}")
            return jsonify({"status": "failed", "message": "Invalid or expired session"})
        
        # Get session and current challenge
        session = active_challenges[session_id]
        current_index = session["current_index"]
        challenge_type = session["sequence"][current_index]["type"]
        
        # Get frames
        frames = session['frames']
        
        logger.info(f"Verifying challenge {current_index + 1}/{len(session['sequence'])}, type: {challenge_type}")
        log_frames_info(frames, "Liveness verification: ")
        
        # Check if enough frames received
        if len(frames) < 4:
            logger.warning(f"Not enough frames for verification. Received: {len(frames)}, need at least 4.")
            return jsonify({
                "status": "failed", 
                "message": f"Not enough frames for verification. Received: {len(frames)}, need at least 4."
            })
        
        # Perform liveness verification based on challenge type
        is_live = False
        liveness_score = 0.0
        details = ""
        
        logger.info(f"Starting {challenge_type} verification...")
        
        # BLINK CHALLENGE
        if challenge_type == 'blink':
            logger.info("Verifying blink challenge...")
            blink_count, ear_values = detect_blinks(frames)
            natural_movement, flow_score, flow_details = analyze_optical_flow(frames)
            
            logger.info(f"Blink detection - Count: {blink_count}, Natural movement: {natural_movement}, Flow score: {flow_score:.2f}")
            
            # Score calculation for blink
            blink_score = min(blink_count / 2.0, 1.0)  # 2 blinks = max score
            movement_score = flow_score
            
            # Combined score
            liveness_score = blink_score * 0.7 + movement_score * 0.3
            
            # Verification criteria
            is_live = (blink_count >= 1 and natural_movement and liveness_score >= 0.5)
            details = f"Blinks: {blink_count}, Natural movement: {natural_movement}, Score: {liveness_score:.2f}"
            
        # SMILE CHALLENGE
        elif challenge_type == 'smile':
            logger.info("Verifying smile challenge...")
            smile_detected, smile_confidence = detect_smile(frames)
            natural_movement, flow_score, flow_details = analyze_optical_flow(frames)
            
            logger.info(f"Smile detection - Detected: {smile_detected}, Confidence: {smile_confidence:.2f}, Natural movement: {natural_movement}")
            
            # Score calculation for smile
            smile_score = smile_confidence
            movement_score = flow_score
            
            # Combined score
            liveness_score = smile_score * 0.7 + movement_score * 0.3
            
            # Verification criteria
            is_live = (smile_detected and natural_movement and liveness_score >= 0.5)
            details = f"Smile detected: {smile_detected}, Confidence: {smile_confidence:.2f}, Natural movement: {natural_movement}, Score: {liveness_score:.2f}"
            
        # HEAD MOVEMENT CHALLENGES
        elif challenge_type in ['head_left', 'head_right']:
            logger.info("Verifying horizontal head movement challenge...")
            movement_detected, movement_range = detect_head_movement(frames, 'horizontal')
            natural_movement, flow_score, flow_details = analyze_optical_flow(frames)
            
            logger.info(f"Head movement - Detected: {movement_detected}, Range: {movement_range}, Natural movement: {natural_movement}")
            
            # Score calculation for head movement
            movement_score = min(movement_range / 40.0, 1.0)  # 40 pixels = max score
            flow_movement_score = flow_score
            
            # Combined score
            liveness_score = movement_score * 0.7 + flow_movement_score * 0.3
            
            # Verification criteria - more lenient
            is_live = (movement_range > 15 and natural_movement and liveness_score >= 0.4)  # Reduced threshold
            details = f"Head movement detected: {movement_detected}, Range: {movement_range}, Natural movement: {natural_movement}, Score: {liveness_score:.2f}"
            
        elif challenge_type in ['head_up', 'head_down']:
            logger.info("Verifying vertical head movement challenge...")
            movement_detected, movement_range = detect_head_movement(frames, 'vertical')
            natural_movement, flow_score, flow_details = analyze_optical_flow(frames)
            
            logger.info(f"Vertical head movement - Detected: {movement_detected}, Range: {movement_range}, Natural movement: {natural_movement}")
            
            # Score calculation for vertical head movement
            movement_score = min(movement_range / 40.0, 1.0)  # 40 pixels = max score
            flow_movement_score = flow_score
            
            # Combined score
            liveness_score = movement_score * 0.7 + flow_movement_score * 0.3
            
            # Verification criteria - more lenient
            is_live = (movement_range > 15 and natural_movement and liveness_score >= 0.4)  # Reduced threshold
            details = f"Vertical movement detected: {movement_detected}, Range: {movement_range}, Natural movement: {natural_movement}, Score: {liveness_score:.2f}"
            
        # GENERAL CHALLENGE (fallback)
        else:
            logger.info("Verifying general liveness challenge...")
            is_live, general_score, general_details = verify_liveness_multimethod(frames, transaction_amount)
            liveness_score = general_score
            details = general_details
        
        # Ensure minimum score
        liveness_score = max(liveness_score, 0.0)
        
        # Store the result of this challenge
        if "challenge_results" not in session:
            session["challenge_results"] = []
        if "liveness_scores" not in session:
            session["liveness_scores"] = []
            
        # Extend lists if needed
        while len(session["challenge_results"]) <= current_index:
            session["challenge_results"].append(False)
        while len(session["liveness_scores"]) <= current_index:
            session["liveness_scores"].append(0.0)
            
        session["challenge_results"][current_index] = is_live
        session["liveness_scores"][current_index] = liveness_score
        
        # Log results
        logger.info(f"Challenge {current_index + 1} verification - Result: {'Success' if is_live else 'Failed'}, Score: {liveness_score:.2f}")
        logger.info(f"Details: {details}")
        
        # Check if this is the final challenge
        is_final = current_index >= len(session["sequence"]) - 1
        
        if is_final and is_live:
            # Mark session as completed and calculate final score
            session["completed"] = True
            session["final_liveness_score"] = sum(session["liveness_scores"]) / len(session["liveness_scores"])
            logger.info(f"All challenges completed. Final liveness score: {session['final_liveness_score']:.2f}")
        
        # Convert numpy types to Python types for JSON serialization
        is_live = bool(is_live)
        liveness_score = float(liveness_score)
        movement_detected = bool(movement_detected) if 'movement_detected' in locals() else None
        natural_movement = bool(natural_movement) if 'natural_movement' in locals() else None
        
        # Return results
        return jsonify({
            "status": "success" if is_live else "failed",
            "liveness_verified": is_live,
            "liveness_score": liveness_score,
            "details": details,
            "frames_processed": len(frames),
            "transaction_amount": float(transaction_amount),
            "threshold_used": float(get_liveness_threshold(transaction_amount)),
            "current_challenge": int(current_index + 1),
            "total_challenges": int(len(session["sequence"])),
            "is_final": is_final,
            "challenge_type": challenge_type
        })
    
    except Exception as e:
        logger.error(f"Error verifying liveness: {str(e)}", exc_info=True)
        return jsonify({"status": "failed", "message": f"Failed to verify liveness: {str(e)}"})

# Register
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        user_id = data['user_id']
        client_ip = request.remote_addr
        
        # Validate user_id format  
        if not user_id:
            logger.warning(f"Missing user_id")
            return jsonify({"status": "failed", "message": "User ID is required."})

        # Convert to string for consistency
        user_id = str(user_id)
        logger.info(f"Processing registration for user_id: {user_id}")
        
        # Check if liveness verification was performed
        liveness_session_id = data.get('liveness_session_id')
        liveness_score = 0.0

        if liveness_session_id:
            if (liveness_session_id in active_challenges and 
                active_challenges[liveness_session_id].get('completed', False)):
                # Use results from completed liveness challenge sequence
                is_live = all(active_challenges[liveness_session_id].get('challenge_results', [False]))
                if not is_live:
                    return jsonify({
                        "status": "failed", 
                        "message": "Liveness verification sequence failed. Please try again."
                    })
                
                # Get liveness score if available
                liveness_score = active_challenges[liveness_session_id].get('final_liveness_score', 0.5)
            else:
                # Liveness challenge sequence not completed or invalid
                logger.warning(f"Invalid or incomplete liveness session sequence: {liveness_session_id}")
                # We'll continue but with a note that liveness wasn't verified
        
        # Check if image data is provided
        if 'image_base64' not in data or not data['image_base64']:
            logger.warning("No image data provided")
            return jsonify({"status": "failed", "message": "No image data provided."})
        
        image = base64_to_image(data['image_base64'])
        if image is None:
            logger.error("Failed to decode base64 image")
            log_security_event(user_id, "register", "failed", client_ip=client_ip)
            return jsonify({"status": "failed", "message": "Invalid base64 image."})
        
        # Extract and process face
        face, error = extract_face(image)
        if error:
            logger.warning(f"Face extraction failed: {error}")
            log_security_event(user_id, "register", "failed", client_ip=client_ip)
            return jsonify({"status": "failed", "message": error})
        
        # Check if user already exists
        db = connect_db()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE user_id = %s", (user_id,))
        existing_count = cursor.fetchone()[0]
        cursor.close()
        db.close()
        
        if existing_count > 0:
            logger.warning(f"User {user_id} already exists")
            log_security_event(user_id, "register", "failed", client_ip=client_ip)
            return jsonify({"status": "failed", "message": "User already exists."})
        
        # Store multiple embeddings for the face
        store_multiple_embeddings(user_id, face, count=15)
        
        # Train or update the SVM model if needed
        if os.path.exists('svm_classifier.pkl'):
            train_svm_classifier()
        
        log_security_event(user_id, "register", "success", client_ip=client_ip, liveness_score=liveness_score)
        return jsonify({
            "status": "success", 
            "message": "User registered successfully",
            "liveness_verified": liveness_session_id is not None,
            "liveness_score": liveness_score
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return jsonify({"status": "failed", "message": f"Registration error: {str(e)}"})

# Recognize
@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        client_ip = request.remote_addr
        
        # Extract user_id for 1:1 verification
        user_id = data.get('user_id')
        
        # Check liveness verification (if available)
        liveness_session_id = data.get('liveness_session_id')
        liveness_score = 0.0

        if liveness_session_id:
            if (liveness_session_id in active_challenges and 
                active_challenges[liveness_session_id].get('completed', False)):
                # Use results from completed liveness challenge sequence
                is_live = all(active_challenges[liveness_session_id].get('challenge_results', [False]))
                if not is_live:
                    return jsonify({
                        "status": "failed", 
                        "message": "Liveness verification sequence failed. Please try again."
                    })
                
                # Get liveness score if available
                liveness_score = active_challenges[liveness_session_id].get('final_liveness_score', 0.5)
            else:
                # Liveness challenge sequence not completed or invalid
                logger.warning(f"Invalid or incomplete liveness session sequence: {liveness_session_id}")
                # Continue but with a note that liveness wasn't verified
        
        # Check if image data is provided
        if 'image_base64' not in data or not data['image_base64']:
            logger.warning("No image data provided")
            return jsonify({"status": "failed", "message": "No image data provided."})
        
        image = base64_to_image(data['image_base64'])
        if image is None:
            logger.error("Failed to decode base64 image")
            log_security_event("unknown", "login", "failed", client_ip=client_ip)
            return jsonify({"status": "failed", "message": "Invalid base64 image."})
        
        # Extract and process face
        face, error = extract_face(image)
        if error:
            logger.warning(f"Face extraction failed: {error}")
            log_security_event("unknown", "login", "failed", client_ip=client_ip)
            return jsonify({"status": "failed", "message": error})
        
        # Get face embedding
        embedding = get_embedding(face)
        
        # Use hybrid recognition system
        recognized_user, similarity, match_count = hybrid_recognition(embedding, user_id)
        
        if recognized_user:
            # Success - user recognized
            logger.info(f"User recognized: {recognized_user}, similarity: {similarity:.4f}, matches: {match_count}")
            log_security_event(recognized_user, "login", "success", 
                              similarity_score=similarity, client_ip=client_ip,
                              liveness_score=liveness_score)
            
            return jsonify({
                "status": "success", 
                "user_id": recognized_user, 
                "similarity": float(similarity),
                "matches": int(match_count),
                "liveness_verified": liveness_session_id is not None,
                "liveness_score": float(liveness_score)
            })
        else:
            # Failed recognition
            logger.warning(f"User not recognized. Best similarity: {similarity:.4f}, matches: {match_count}")
            log_security_event(user_id if user_id else "unknown", "login", "failed", 
                             similarity_score=similarity, client_ip=client_ip,
                             liveness_score=liveness_score)
            
            return jsonify({
                "status": "failed", 
                "message": "Face not recognized or confidence too low.",
                "similarity": float(similarity),
                "matches": int(match_count),
                "liveness_verified": liveness_session_id is not None,
                "liveness_score": float(liveness_score)
            })
            
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}", exc_info=True)
        log_security_event("unknown", "login", "error", client_ip=request.remote_addr)
        return jsonify({"status": "failed", "message": f"Recognition error: {str(e)}"})
    
    # Health endpoint
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check database connection
        db = connect_db()
        db.close()
        
        # Check if models are loaded
        detector_loaded = detector is not None
        facenet_loaded = facenet_model is not None
        dlib_loaded = face_landmark_detector is not None
        
        status = "healthy" if detector_loaded and facenet_loaded and dlib_loaded else "degraded"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "connected",
                "detector": "loaded" if detector_loaded else "error",
                "facenet": "loaded" if facenet_loaded else "error",
                "dlib": "loaded" if dlib_loaded else "error"
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Modified /authenticate endpoint
@app.route('/authenticate', methods=['POST'])
def authenticate():
    """
    Combined endpoint for liveness detection and recognition.
    Supports both session-based liveness verification (recommended) and real-time verification (legacy).
    NEW: Supports security_only mode for registration purposes.
    """
    try:
        data = request.json
        client_ip = request.remote_addr
        
        # NEW: Check if this is security-only verification (for registration)
        security_only = data.get('security_only', False)
        
        # Extract user_id for 1:1 verification (skip if security_only)
        user_id = data.get('user_id') if not security_only else None
        
        # Get challenge type for specific verification
        challenge_type = data.get('challenge_type', 'general')
        
        # Get transaction amount
        transaction_amount = data.get('transaction_amount', 0)
        
        # Check if using session-based liveness verification (NEW SECURE APPROACH)
        liveness_session_id = data.get('liveness_session_id')
        
        if liveness_session_id:
            # SESSION-BASED LIVENESS VERIFICATION (Secure against replay attacks)
            logger.info(f"Using session-based liveness verification: {liveness_session_id}")
            logger.info(f"Authentication request - User: {user_id if user_id else 'unknown'}, Session: {liveness_session_id}, Transaction: {transaction_amount}")
            
            if (liveness_session_id in active_challenges and 
                active_challenges[liveness_session_id].get('completed', False)):
                # Use results from completed liveness challenge sequence
                challenge_results = active_challenges[liveness_session_id].get('challenge_results', [False])
                is_live = all(challenge_results)
                
                logger.info(f"Liveness session results: {challenge_results}, All passed: {is_live}")
                
                if not is_live:
                    logger.warning("Liveness verification sequence failed")
                    log_security_event(user_id if user_id else "unknown", "login", "failed_liveness_sequence", client_ip=client_ip)
                    return jsonify({
                        "status": "failed", 
                        "message": "Liveness verification sequence failed. Please try again."
                    })
                
                # Get liveness score if available
                liveness_score = active_challenges[liveness_session_id].get('final_liveness_score', 0.5)
                logger.info(f"Using liveness score from session: {liveness_score:.2f}")
                
                # Check if image data is provided for face recognition
                if 'image_base64' not in data or not data['image_base64']:
                    logger.warning("No image data provided for face recognition")
                    return jsonify({"status": "failed", "message": "No image data provided for face recognition."})
                
                # Process main image for face recognition only
                logger.info("Processing image for face recognition...")
                main_image = base64_to_image(data['image_base64'])
                if main_image is None:
                    logger.error("Failed to decode main image")
                    log_security_event("unknown", "login", "failed", client_ip=client_ip)
                    return jsonify({"status": "failed", "message": "Invalid main image data."})
                
                # Log image resolution
                if main_image is not None:
                    height, width = main_image.shape[:2]
                    logger.info(f"Main image resolution: {width}x{height}")
                
                # Extract face from main image for recognition
                logger.info("Extracting face from main image...")
                face, error = extract_face(main_image)
                if error:
                    logger.warning(f"Face extraction failed: {error}")
                    log_security_event("unknown", "login", "failed", client_ip=client_ip)
                    return jsonify({"status": "failed", "message": error})
                
                # Log face extraction success
                if face is not None:
                    height, width = face.shape[:2]
                    logger.info(f"Extracted face resolution: {width}x{height}")
                
                # Get face embedding for recognition
                logger.info("Generating face embedding...")
                embedding = get_embedding(face)
                
                # NEW: Check if security_only mode
                if security_only:
                    # For registration: return security verification only (no face recognition)
                    logger.info("Security-only mode: Skipping face recognition for registration")
                    return jsonify({
                        "status": "success",
                        "liveness_verified": True,
                        "liveness_score": float(liveness_score),
                        "security_only": True,
                        "message": "Security verification passed - ready for registration",
                        "verification_method": "session_based"
                    })
                
                # Perform recognition (skip liveness since it was already verified)
                logger.info(f"Starting facial recognition for user: {user_id if user_id else 'unknown'}")
                recognized_user, similarity, match_count = hybrid_recognition(embedding, user_id)
                logger.info(f"Recognition result - User: {recognized_user if recognized_user else 'unknown'}, Similarity: {similarity:.4f}, Matches: {match_count}")
                
                if recognized_user:
                    # Success - user recognized
                    logger.info(f"User authenticated: {recognized_user}, similarity: {similarity:.4f}, matches: {match_count}, transaction_amount: {transaction_amount}")
                    log_security_event(recognized_user, "login", "success",
                                      similarity_score=similarity, client_ip=client_ip,
                                      liveness_score=liveness_score)

                    return jsonify({
                        "status": "success",
                        "user_id": recognized_user,
                        "similarity": float(similarity),
                        "matches": int(match_count),
                        "liveness_verified": True,
                        "liveness_score": float(liveness_score),
                        "details": "Authentication successful: liveness verified and user recognized.",
                        "verification_method": "session_based"
                    })
                else:
                    # Failed recognition
                    logger.warning(f"User not recognized. Best similarity: {similarity:.4f}, matches: {match_count}, transaction_amount: {transaction_amount}")
                    log_security_event(user_id if user_id else "unknown", "login", "failed",
                                     similarity_score=similarity, client_ip=client_ip,
                                     liveness_score=liveness_score)

                    return jsonify({
                        "status": "failed",
                        "message": "Face not recognized or confidence too low.",
                        "similarity": float(similarity),
                        "matches": int(match_count),
                        "liveness_verified": True,
                        "liveness_score": float(liveness_score),
                        "verification_method": "session_based"
                    })
            else:
                # Liveness challenge sequence not completed or invalid
                logger.warning(f"Invalid or incomplete liveness session sequence: {liveness_session_id}")
                
                # Check if session exists but incomplete
                if liveness_session_id in active_challenges:
                    session = active_challenges[liveness_session_id]
                    current_challenge = session.get('current_index', 0) + 1
                    total_challenges = len(session.get('sequence', []))
                    
                    return jsonify({
                        "status": "failed", 
                        "message": f"Liveness verification incomplete. Please complete challenge {current_challenge} of {total_challenges}.",
                        "liveness_session_status": "incomplete",
                        "current_challenge": current_challenge,
                        "total_challenges": total_challenges
                    })
                else:
                    return jsonify({
                        "status": "failed", 
                        "message": "Invalid liveness session. Please start a new liveness verification.",
                        "liveness_session_status": "invalid"
                    })
        
        # REAL-TIME LIVENESS VERIFICATION (Legacy approach - less secure)
        logger.info(f"Using real-time liveness verification - User: {user_id if user_id else 'unknown'}, Challenge: {challenge_type}, Transaction: {transaction_amount}, Security-only: {security_only}")
        if not security_only:
            logger.warning("WARNING: Using less secure real-time liveness verification. Consider using session-based approach for better security.")
        
        # Generate a random challenge (blink, smile, head movement)
        if challenge_type == 'random':
            challenge_types = ['blink', 'smile', 'head_left', 'head_right']
            challenge_type = random.choice(challenge_types)
            logger.info(f"Generated random challenge: {challenge_type}")
        
        # Check if image data is provided
        if 'image_base64' not in data or not data['image_base64']:
            logger.warning("No image data provided")
            return jsonify({"status": "failed", "message": "No image data provided."})

        # Get additional frames for liveness detection
        additional_frames = data.get('additional_frames', [])

        # Log number of frames for debugging
        logger.info(f"Number of frames received: {len(additional_frames) + 1}, transaction_amount: {transaction_amount}")

        # Increased minimum additional frames requirement to 5
        if len(additional_frames) < 5:
            logger.warning("Not enough frames for liveness detection")
            return jsonify({
                "status": "failed",
                "message": "At least 5 additional frames are required for liveness detection."
            })

        # Process main image
        main_image = base64_to_image(data['image_base64'])
        if main_image is None:
            logger.error("Failed to decode main image")
            log_security_event("unknown", "login", "failed", client_ip=client_ip)
            return jsonify({"status": "failed", "message": "Invalid main image data."})

        # Process additional frames
        frames = [main_image]
        valid_frames = 1
        invalid_frames = 0
        
        for i, frame_base64 in enumerate(additional_frames):
            frame = base64_to_image(frame_base64)
            if frame is not None:
                frames.append(frame)
                valid_frames += 1
            else:
                invalid_frames += 1
                logger.warning(f"Failed to decode additional frame {i+1}")
                
        logger.info(f"Frame processing summary - Valid: {valid_frames}, Invalid: {invalid_frames}, Total: {len(frames)}")

        # Increased minimum frames from 4 to 6
        if len(frames) < 6:  # Main + at least 5 additional
            logger.warning("Not enough valid frames for liveness detection")
            return jsonify({
                "status": "failed",
                "message": "Not enough valid frames for liveness detection."
            })

        # Log resolution of main image
        if main_image is not None:
            height, width = main_image.shape[:2]
            logger.info(f"Main image resolution: {width}x{height}")

        # Check for static image (all frames identical)
        logger.info("Checking for static image...")
        if detect_static_image(frames):
            logger.warning("Static image detected - frames are too similar")
            log_security_event(user_id if user_id else "unknown", "login", "failed_static_image", client_ip=client_ip)
            return jsonify({
                "status": "failed",
                "message": "Static image detected. Please provide a live video feed.",
                "liveness_verified": False,
                "liveness_score": 0.0,
                "verification_method": "real_time"
            })

        # Extract face from main image for recognition
        logger.info("Extracting face from main image...")
        face, error = extract_face(main_image)
        if error:
            logger.warning(f"Face extraction failed: {error}")
            log_security_event("unknown", "login", "failed", client_ip=client_ip)
            return jsonify({"status": "failed", "message": error})

        # Log face extraction success
        if face is not None:
            height, width = face.shape[:2]
            logger.info(f"Extracted face resolution: {width}x{height}")

        # Get face embedding for recognition (needed for both modes)
        logger.info("Generating face embedding...")
        embedding = get_embedding(face)

        # Log challenge type
        logger.info(f"Starting liveness verification with challenge type: {challenge_type}")

        # Perform enhanced liveness detection with transaction amount
        logger.info(f"Running multi-method liveness verification for transaction amount: {transaction_amount}")
        is_live, liveness_score, liveness_details = verify_liveness_multimethod(frames, transaction_amount)
        logger.info(f"Multi-method verification result: {'Success' if is_live else 'Failed'}, Score: {liveness_score:.2f}, Details: {liveness_details}")

        # Specific challenge verification based on challenge_type
        logger.info(f"Running specific challenge verification for: {challenge_type}")
        if challenge_type == 'blink':
            logger.info("Verifying blink challenge...")
            blink_count, _ = detect_blinks(frames)
            logger.info(f"Blink challenge verification - Blinks detected: {blink_count}")
            if blink_count < 1:
                logger.warning("Blink challenge failed - no blinks detected")
                return jsonify({
                    "status": "failed",
                    "message": "Blink challenge failed. No blinks detected.",
                    "liveness_verified": False,
                    "liveness_score": float(liveness_score),
                    "verification_method": "real_time"
                })
        elif challenge_type == 'smile':
            logger.info("Verifying smile challenge...")
            smile_detected, smile_confidence = detect_smile(frames)
            logger.info(f"Smile challenge verification - Smile detected: {smile_detected}, Confidence: {smile_confidence:.2f}")
            if not smile_detected:
                logger.warning("Smile challenge failed - no smile detected")
                return jsonify({
                    "status": "failed",
                    "message": "Smile challenge failed. No smile detected.",
                    "liveness_verified": False,
                    "liveness_score": float(liveness_score),
                    "verification_method": "real_time"
                })
        elif challenge_type in ['head_left', 'head_right', 'head_movement']:
            logger.info("Verifying horizontal head movement challenge...")
            movement_detected, movement_range = detect_head_movement(frames, 'horizontal')
            logger.info(f"Head movement challenge verification - Movement detected: {movement_detected}, Range: {movement_range}")
            if not movement_detected:
                logger.warning("Head movement challenge failed - no movement detected")
                return jsonify({
                    "status": "failed",
                    "message": "Head movement challenge failed. No significant movement detected.",
                    "liveness_verified": False,
                    "liveness_score": float(liveness_score),
                    "verification_method": "real_time"
                })
        elif challenge_type in ['head_up', 'head_down']:
            logger.info("Verifying vertical head movement challenge...")
            movement_detected, movement_range = detect_head_movement(frames, 'vertical')
            logger.info(f"Vertical head movement verification - Movement detected: {movement_detected}, Range: {movement_range}")
            if not movement_detected:
                logger.warning("Head movement challenge failed - no vertical movement detected")
                return jsonify({
                    "status": "failed",
                    "message": "Head movement challenge failed. No significant vertical movement detected.",
                    "liveness_verified": False,
                    "liveness_score": float(liveness_score),
                    "verification_method": "real_time"
                })

        # Mandatory check for natural movement via optical flow
        logger.info("Performing optical flow analysis for natural movement...")
        natural_movement, flow_score, flow_details = analyze_optical_flow(frames)
        logger.info(f"Optical flow result: {'Natural' if natural_movement else 'Suspicious'}, Score: {flow_score:.2f}, Details: {flow_details}")
        if not natural_movement:
            logger.warning(f"No natural movement detected: {flow_details}")
            return jsonify({
                "status": "failed",
                "message": f"Liveness check failed. No natural movement detected - possible static image or video replay.",
                "liveness_verified": False,
                "liveness_score": float(liveness_score),
                "verification_method": "real_time"
            })

        # If liveness check passed, handle based on mode
        if is_live:
            # NEW: Check if security_only mode for registration
            if security_only:
                logger.info("Security-only mode: Liveness verification passed for registration")
                return jsonify({
                    "status": "success",
                    "liveness_verified": True,
                    "liveness_score": float(liveness_score),
                    "security_only": True,
                    "message": "Security verification passed - ready for registration",
                    "verification_method": "real_time"
                })
            
            # Use hybrid recognition system for authentication
            logger.info(f"Starting facial recognition for user: {user_id if user_id else 'unknown'}")
            recognized_user, similarity, match_count = hybrid_recognition(embedding, user_id)
            logger.info(f"Recognition result - User: {recognized_user if recognized_user else 'unknown'}, Similarity: {similarity:.4f}, Matches: {match_count}")

            if recognized_user:
                # Success - user recognized
                logger.info(f"User authenticated: {recognized_user}, similarity: {similarity:.4f}, matches: {match_count}, transaction_amount: {transaction_amount}")
                log_security_event(recognized_user, "login", "success",
                                  similarity_score=similarity, client_ip=client_ip,
                                  liveness_score=liveness_score)

                return jsonify({
                    "status": "success",
                    "user_id": recognized_user,
                    "similarity": float(similarity),
                    "matches": int(match_count),
                    "liveness_verified": True,
                    "liveness_score": float(liveness_score),
                    "details": "Authentication successful: liveness verified and user recognized.",
                    "verification_method": "real_time"
                })
            else:
                # Failed recognition
                logger.warning(f"User not recognized. Best similarity: {similarity:.4f}, matches: {match_count}, transaction_amount: {transaction_amount}")
                log_security_event(user_id if user_id else "unknown", "login", "failed",
                                 similarity_score=similarity, client_ip=client_ip,
                                 liveness_score=liveness_score)

                return jsonify({
                    "status": "failed",
                    "message": "Face not recognized or confidence too low.",
                    "similarity": float(similarity),
                    "matches": int(match_count),
                    "liveness_verified": True,
                    "liveness_score": float(liveness_score),
                    "verification_method": "real_time"
                })
        else:
            # Failed liveness check
            logger.warning(f"Liveness check failed. Details: {liveness_details}, transaction_amount: {transaction_amount}")
            log_security_event(user_id if user_id else "unknown", "login", "failed",
                             client_ip=client_ip, liveness_score=liveness_score)

            return jsonify({
                "status": "failed",
                "message": f"Liveness check failed. {liveness_details}",
                "liveness_verified": False,
                "liveness_score": float(liveness_score),
                "verification_method": "real_time"
            })

    except Exception as e:
        logger.error(f"Authentication error: {str(e)}", exc_info=True)
        log_security_event("unknown", "login", "error", client_ip=request.remote_addr)
        return jsonify({"status": "failed", "message": f"Authentication error: {str(e)}"})
    
if __name__ == '__main__':
    setup_database()
    app.run(host='0.0.0.0', port=5000, debug=True)
