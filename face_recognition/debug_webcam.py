import cv2
import numpy as np
import dlib
import argparse
import time
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Constants - adjust these to match your face_api.py settings
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES_THRESHOLD = 2

# Load models - ensure these paths match what you have in face_api.py
try:
    face_detector = dlib.get_frontal_face_detector()
    face_landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    logger.info("Please download the shape predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit(1)

# Helper functions (copy from your face_api.py)
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    try:
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye[0] - eye[3])
        
        # Avoid division by zero
        if C == 0:
            return 0.0
            
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        logger.warning(f"Error calculating EAR: {e}")
        return 0.0

def mouth_aspect_ratio(mouth):
    try:
        # Compute the euclidean distances 
        horizontal = np.linalg.norm(mouth[0] - mouth[6])  # width of mouth
        vertical = np.linalg.norm(mouth[3] - mouth[9])    # height of mouth
        
        # Avoid division by zero
        if horizontal == 0:
            return 0.0
            
        # Compute the mouth aspect ratio
        mar = vertical / horizontal
        return mar
    except Exception as e:
        logger.warning(f"Error calculating MAR: {e}")
        return 0.0

# Function to draw debug information on frame
def draw_debug_info(frame, detection_results):
    """
    Draw debug information on the frame to visualize what's being detected
    """
    # Create a copy of the frame to avoid modifying the original
    debug_frame = frame.copy()
    
    # Add text with white color and black outline for readability
    def add_text(text, position, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_frame, text, position, font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(debug_frame, text, position, font, 0.6, color, 1, cv2.LINE_AA)
    
    # Draw detection results
    y_pos = 30
    add_text(f"Challenge: {detection_results.get('challenge_type', 'unknown')}", (10, y_pos))
    y_pos += 25
    
    # Draw face detection status
    face_detected = detection_results.get('face_detected', False)
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    add_text(f"Face detected: {face_detected}", (10, y_pos), face_color)
    y_pos += 25
    
    # Draw eye status for blink detection
    if 'ear' in detection_results:
        ear = detection_results.get('ear', 0)
        threshold = detection_results.get('ear_threshold', EAR_THRESHOLD)
        is_blinking = ear < threshold
        ear_color = (0, 0, 255) if is_blinking else (0, 255, 0)
        add_text(f"EAR: {ear:.3f} (Threshold: {threshold:.3f})", (10, y_pos), ear_color)
        y_pos += 25
        add_text(f"Blinks detected: {detection_results.get('blink_count', 0)}", (10, y_pos))
        y_pos += 25
    
    # Draw mouth status for smile detection
    if 'mar' in detection_results:
        mar = detection_results.get('mar', 0)
        mar_color = (0, 255, 0) if detection_results.get('smile_detected', False) else (0, 255, 255)
        add_text(f"MAR: {mar:.3f}", (10, y_pos), mar_color)
        y_pos += 25
        add_text(f"Smile detected: {detection_results.get('smile_detected', False)}", (10, y_pos))
        y_pos += 25
    
    # Draw head movement info
    if 'movement_range' in detection_results:
        movement_range = detection_results.get('movement_range', 0)
        movement_type = detection_results.get('movement_type', 'horizontal')
        movement_detected = detection_results.get('movement_detected', False)
        movement_color = (0, 255, 0) if movement_detected else (0, 255, 255)
        add_text(f"{movement_type.capitalize()} movement: {movement_range:.1f} px", (10, y_pos), movement_color)
        y_pos += 25
    
    # Draw overall liveness status
    liveness_verified = detection_results.get('liveness_verified', False)
    liveness_score = detection_results.get('liveness_score', 0.0)
    liveness_color = (0, 255, 0) if liveness_verified else (0, 0, 255)
    add_text(f"Liveness verified: {liveness_verified}", (10, y_pos), liveness_color)
    y_pos += 25
    add_text(f"Liveness score: {liveness_score:.2f}", (10, y_pos))
    
    # Draw a background box for challenge instructions
    instruction = detection_results.get('instruction', '')
    if instruction:
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(debug_frame, 
                     (10, debug_frame.shape[0] - 50), 
                     (20 + text_size[0], debug_frame.shape[0] - 20), 
                     (0, 0, 0), -1)
        add_text(instruction, (15, debug_frame.shape[0] - 30), (255, 255, 255))
    
    # Draw face landmarks if available
    if 'landmarks' in detection_results:
        landmarks = detection_results.get('landmarks', [])
        for (x, y) in landmarks:
            cv2.circle(debug_frame, (x, y), 2, (0, 255, 0), -1)
    
    # Draw eye contours if available
    if 'left_eye' in detection_results and 'right_eye' in detection_results:
        for eye in [detection_results['left_eye'], detection_results['right_eye']]:
            hull = cv2.convexHull(np.array(eye))
            cv2.drawContours(debug_frame, [hull], -1, (0, 255, 255), 1)
    
    return debug_frame

# Function to process the frame for different challenge types
def process_frame(frame, challenge_type, results, frame_history):
    """
    Process the frame for different liveness challenge types and update results dictionary
    """
    # Define eye and mouth landmark indices
    (lStart, lEnd) = (42, 48)  # Left eye landmarks
    (rStart, rEnd) = (36, 42)  # Right eye landmarks
    mouth_indices = list(range(48, 68))  # Mouth landmarks
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    rects = face_detector(gray, 0)
    results['face_detected'] = len(rects) > 0
    
    if results['face_detected']:
        # Get facial landmarks
        shape = face_landmark_detector(gray, rects[0])
        landmarks = shape_to_np(shape)
        results['landmarks'] = landmarks
        
        # Extract eye coordinates
        left_eye = landmarks[lStart:lEnd]
        right_eye = landmarks[rStart:rEnd]
        results['left_eye'] = left_eye
        results['right_eye'] = right_eye
        
        # Extract mouth coordinates
        mouth = landmarks[mouth_indices]
        results['mouth'] = mouth
        
        # Process based on challenge type
        if challenge_type == 'blink':
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            results['ear'] = ear
            results['ear_threshold'] = EAR_THRESHOLD
            
            # Check for blink
            if ear < EAR_THRESHOLD:
                results['blink_counter'] += 1
            else:
                if results['blink_counter'] >= CONSECUTIVE_FRAMES_THRESHOLD:
                    results['blink_count'] += 1
                    logger.info(f"Blink detected! EAR: {ear:.3f}")
                results['blink_counter'] = 0
            
            # Update liveness status based on blinks
            results['liveness_verified'] = results['blink_count'] >= 1
            results['liveness_score'] = min(results['blink_count'] / 2.0, 1.0)
            
        elif challenge_type == 'smile':
            # Calculate MAR
            mar = mouth_aspect_ratio(mouth)
            results['mar'] = mar
            
            # Calculate width ratio
            face_width = rects[0].right() - rects[0].left()
            mouth_width = mouth[6][0] - mouth[0][0]
            width_ratio = mouth_width / face_width
            results['width_ratio'] = width_ratio
            
            # Check for smile
            smile_detected = mar < 0.5 and width_ratio > 0.4
            
            if smile_detected:
                results['smile_confidence'] = max(results['smile_confidence'], 0.7)
                if not results['smile_detected']:
                    logger.info(f"Smile detected! MAR: {mar:.3f}, Width ratio: {width_ratio:.3f}")
            
            # Store MAR values for smile animation detection
            results['mar_values'].append(mar)
            if len(results['mar_values']) > 20:  # Keep only recent values
                results['mar_values'].pop(0)
                
            # Calculate MAR range for smile animation detection
            if len(results['mar_values']) > 3:
                mar_min = min(results['mar_values'])
                mar_max = max(results['mar_values'])
                mar_range = mar_max - mar_min
                
                # Check for significant change in MAR (smile animation)
                if mar_range > 0.15:
                    results['smile_detected'] = True
                    results['smile_confidence'] = max(results['smile_confidence'], 0.9)
                    logger.info(f"Smile animation detected! MAR range: {mar_range:.3f}")
                elif mar_min < 0.5 and sum(results['mar_values']) / len(results['mar_values']) < 0.6:
                    # Sustained smile
                    results['smile_detected'] = True
                    results['smile_confidence'] = max(results['smile_confidence'], 0.8)
            
            # Update liveness status based on smile
            results['liveness_verified'] = results['smile_detected']
            results['liveness_score'] = results['smile_confidence']
            
        elif challenge_type in ['head_left', 'head_right', 'head_movement']:
            # Use nose tip (point 30) as reference for head position
            nose_tip = landmarks[30]
            
            # Add to history
            results['nose_positions'].append(nose_tip)
            if len(results['nose_positions']) > 30:  # Keep only recent positions
                results['nose_positions'].pop(0)
                
            # Calculate movement range
            if len(results['nose_positions']) > 3:
                x_positions = [pos[0] for pos in results['nose_positions']]
                x_range = max(x_positions) - min(x_positions)
                results['movement_range'] = x_range
                results['movement_type'] = 'horizontal'
                
                # Check for sufficient movement
                results['movement_detected'] = x_range > 25
                if results['movement_detected'] and not results.get('movement_logged', False):
                    logger.info(f"Horizontal head movement detected! Range: {x_range}")
                    results['movement_logged'] = True
                    
                # Update liveness status based on head movement
                results['liveness_verified'] = results['movement_detected']
                results['liveness_score'] = min(x_range / 50.0, 1.0)
                
        elif challenge_type in ['head_up', 'head_down']:
            # Use nose tip (point 30) as reference for head position
            nose_tip = landmarks[30]
            
            # Add to history
            results['nose_positions'].append(nose_tip)
            if len(results['nose_positions']) > 30:  # Keep only recent positions
                results['nose_positions'].pop(0)
                
            # Calculate movement range
            if len(results['nose_positions']) > 3:
                y_positions = [pos[1] for pos in results['nose_positions']]
                y_range = max(y_positions) - min(y_positions)
                results['movement_range'] = y_range
                results['movement_type'] = 'vertical'
                
                # Check for sufficient movement
                results['movement_detected'] = y_range > 25
                if results['movement_detected'] and not results.get('movement_logged', False):
                    logger.info(f"Vertical head movement detected! Range: {y_range}")
                    results['movement_logged'] = True
                    
                # Update liveness status based on head movement
                results['liveness_verified'] = results['movement_detected']
                results['liveness_score'] = min(y_range / 50.0, 1.0)
    
    # Add frame to history
    frame_history.append(frame.copy())
    if len(frame_history) > 30:  # Keep only recent frames
        frame_history.pop(0)
        
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Debug Mode for Face Liveness Verification')
    parser.add_argument('--challenge', type=str, default='blink',
                       choices=['blink', 'smile', 'head_left', 'head_right', 'head_up', 'head_down'],
                       help='Challenge type to test')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    args = parser.parse_args()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    
    # Set higher resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        logger.error("Error: Could not open camera.")
        return
    
    # Print instructions
    logger.info(f"Starting debug mode for {args.challenge} challenge")
    logger.info("Press 'r' to reset the detection stats")
    logger.info("Press 'q' to quit")
    
    # Initialize results dictionary based on challenge type
    results = {
        'challenge_type': args.challenge,
        'face_detected': False,
        'liveness_verified': False,
        'liveness_score': 0.0
    }
    
    # Add challenge-specific instructions
    if args.challenge == 'blink':
        results['instruction'] = "Please blink naturally 2-3 times"
        results['blink_count'] = 0
        results['blink_counter'] = 0
        
    elif args.challenge == 'smile':
        results['instruction'] = "Please smile naturally"
        results['smile_detected'] = False
        results['smile_confidence'] = 0.0
        results['mar_values'] = []
        
    elif args.challenge in ['head_left', 'head_right', 'head_movement']:
        results['instruction'] = "Please slowly turn your head left and right"
        results['nose_positions'] = []
        results['movement_detected'] = False
        
    elif args.challenge in ['head_up', 'head_down']:
        results['instruction'] = "Please slowly nod your head up and down"
        results['nose_positions'] = []
        results['movement_detected'] = False
    
    # Frame history for temporal analysis
    frame_history = []
    
    # Main loop
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Failed to capture frame.")
            break
        
        # Process frame based on challenge type
        results = process_frame(frame, args.challenge, results, frame_history)
        
        # Draw debug information
        debug_frame = draw_debug_info(frame, results)
        
        # Display the frame
        cv2.imshow('Face Liveness Debug', debug_frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset stats
            logger.info("Resetting detection stats")
            if args.challenge == 'blink':
                results['blink_count'] = 0
                results['blink_counter'] = 0
            elif args.challenge == 'smile':
                results['smile_detected'] = False
                results['smile_confidence'] = 0.0
                results['mar_values'] = []
            elif args.challenge in ['head_left', 'head_right', 'head_movement', 'head_up', 'head_down']:
                results['nose_positions'] = []
                results['movement_detected'] = False
                results['movement_logged'] = False
            
            results['liveness_verified'] = False
            results['liveness_score'] = 0.0
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()