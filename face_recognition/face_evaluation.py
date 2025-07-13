import numpy as np
import cv2
import os
import json
import requests
import base64
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

class FaceRecognitionEvaluator:
    def __init__(self, api_url="http://localhost:5000"):
        """
        Initialize evaluator with Flask API URL (no model instance needed)
        """
        self.api_url = api_url
        self.results = {}
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                # logging.FileHandler('evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_api_connection(self):
        """Check if Flask API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ API connection successful")
                return True
            else:
                self.logger.error(f"❌ API responded with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Cannot connect to API: {e}")
            self.logger.error(f"Make sure your Flask app is running on {self.api_url}")
            return False
    
    def call_recognize_api(self, image_path, user_id=None):
        """Call the /recognize endpoint with an image"""
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
                img_b64 = f"data:image/jpeg;base64,{img_data}"
            
            # Prepare request
            payload = {"image_base64": img_b64}
            if user_id:
                payload["user_id"] = user_id
            
            # Call API
            response = requests.post(f"{self.api_url}/recognize", json=payload)
            result = response.json()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calling API for {image_path}: {e}")
            return {"status": "failed", "message": str(e)}
    
    def call_authenticate_api(self, frames_paths, user_id=None, challenge_type='general'):
        """Call the /authenticate endpoint with video frames"""
        try:
            # Load main image
            with open(frames_paths[0], 'rb') as f:
                main_img_data = base64.b64encode(f.read()).decode('utf-8')
                main_img_b64 = f"data:image/jpeg;base64,{main_img_data}"
            
            # Load additional frames
            additional_frames = []
            for frame_path in frames_paths[1:6]:  # Use up to 5 additional frames
                try:
                    with open(frame_path, 'rb') as f:
                        frame_data = base64.b64encode(f.read()).decode('utf-8')
                        frame_b64 = f"data:image/jpeg;base64,{frame_data}"
                        additional_frames.append(frame_b64)
                except:
                    continue
            
            # Prepare request
            payload = {
                "image_base64": main_img_b64,
                "additional_frames": additional_frames,
                "challenge_type": challenge_type
            }
            if user_id:
                payload["user_id"] = user_id
            
            # Call API
            response = requests.post(f"{self.api_url}/authenticate", json=payload)
            result = response.json()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calling authenticate API: {e}")
            return {"status": "failed", "message": str(e)}
    
    def evaluate_face_recognition(self, test_data_path, ground_truth_labels):
        """
        Evaluate face recognition performance using /recognize endpoint
        
        Args:
            test_data_path: List of paths to test images
            ground_truth_labels: True labels for test images
        """
        if not self.check_api_connection():
            return None
            
        self.logger.info("Starting face recognition evaluation...")
        
        predictions = []
        true_labels = []
        similarities = []
        
        for image_path, true_label in zip(test_data_path, ground_truth_labels):
            self.logger.info(f"Testing {os.path.basename(image_path)} (expected: {true_label})")
            
            result = self.call_recognize_api(image_path, true_label)
            
            if result["status"] == "success":
                predicted_user = result["user_id"]
                similarity = result.get("similarity", 0.0)
                self.logger.info(f"  Predicted: {predicted_user}, Similarity: {similarity:.3f}")
            else:
                predicted_user = "unknown"
                similarity = result.get("similarity", 0.0)
                self.logger.info(f"  Predicted: REJECTED, Similarity: {similarity:.3f}")
            
            predictions.append(predicted_user)
            true_labels.append(true_label)
            similarities.append(similarity)
            
            time.sleep(0.1)  # Small delay to not overwhelm API
        
        # Calculate metrics using your methodology formulas
        # Convert to binary classification for each user (one-vs-all)
        unique_users = list(set(true_labels))
        all_tp = all_tn = all_fp = all_fn = 0
        
        for user in unique_users:
            y_true = [1 if label == user else 0 for label in true_labels]
            y_pred = [1 if pred == user else 0 for pred in predictions]
            
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            
            all_tp += tp
            all_tn += tn
            all_fp += fp
            all_fn += fn
        
        # Calculate final metrics using your methodology formulas
        accuracy = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn) if (all_tp + all_tn + all_fp + all_fn) > 0 else 0
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results
        self.results['face_recognition'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'total_samples': len(true_labels),
            'correct_predictions': sum(1 for t, p in zip(true_labels, predictions) if t == p),
            'tp': all_tp, 'tn': all_tn, 'fp': all_fp, 'fn': all_fn
        }
        
        self.logger.info(f"Face Recognition Results:")
        self.logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        self.logger.info(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        self.logger.info(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        
        return self.results['face_recognition']
    
    def evaluate_liveness_detection(self, live_video_frames, spoof_video_frames):
        """
        Evaluate liveness detection performance using /authenticate endpoint
        
        Args:
            live_video_frames: List of lists containing paths to frames from genuine videos
            spoof_video_frames: List of lists containing paths to frames from spoofed videos
        """
        if not self.check_api_connection():
            return None
            
        self.logger.info("Starting liveness detection evaluation...")
        
        predictions = []
        true_labels = []
        liveness_scores = []
        
        # Test live videos (label = 1)
        for i, frames in enumerate(live_video_frames):
            self.logger.info(f"Testing live video {i+1}/{len(live_video_frames)}")
            
            result = self.call_authenticate_api(frames, challenge_type='general')
            
            if result["status"] == "success" and result.get("liveness_verified", False):
                predictions.append(1)
                liveness_score = result.get("liveness_score", 0.0)
            else:
                predictions.append(0)
                liveness_score = result.get("liveness_score", 0.0)
            
            true_labels.append(1)  # Live
            liveness_scores.append(liveness_score)
            
            time.sleep(0.2)  # Longer delay for liveness detection
        
        # Test spoof videos (label = 0)
        for i, frames in enumerate(spoof_video_frames):
            self.logger.info(f"Testing spoof video {i+1}/{len(spoof_video_frames)}")
            
            result = self.call_authenticate_api(frames, challenge_type='general')
            
            if result["status"] == "success" and result.get("liveness_verified", False):
                predictions.append(1)
                liveness_score = result.get("liveness_score", 0.0)
            else:
                predictions.append(0)
                liveness_score = result.get("liveness_score", 0.0)
            
            true_labels.append(0)  # Spoof
            liveness_scores.append(liveness_score)
            
            time.sleep(0.2)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(true_labels, liveness_scores)
        except:
            auc_roc = 0.0
        
        # Store results
        self.results['liveness_detection'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'mean_liveness_score': np.mean(liveness_scores),
            'total_samples': len(true_labels),
            'live_samples': sum(true_labels),
            'spoof_samples': len(true_labels) - sum(true_labels)
        }
        
        self.logger.info(f"Liveness Detection Results:")
        self.logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        self.logger.info(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        self.logger.info(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        self.logger.info(f"AUC-ROC: {auc_roc:.4f}")
        
        return self.results['liveness_detection']
    
    def load_test_data_from_directory(self, test_dir="test_images"):
        """
        Load test data from directory structure:
        test_images/
        ├── USER_ID1/
        │   ├── test1.jpg
        │   └── test2.jpg
        └── USER_ID2/
            ├── test1.jpg
            └── test2.jpg
        """
        test_images = []
        test_labels = []
        
        if not os.path.exists(test_dir):
            self.logger.error(f"Test directory {test_dir} not found!")
            return test_images, test_labels
        
        users = [d for d in os.listdir(test_dir) 
                if os.path.isdir(os.path.join(test_dir, d))]
        
        for user_id in users:
            user_dir = os.path.join(test_dir, user_id)
            images = [f for f in os.listdir(user_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img in images:
                test_images.append(os.path.join(user_dir, img))
                test_labels.append(user_id)
        
        self.logger.info(f"Loaded {len(test_images)} test images from {len(users)} users")
        return test_images, test_labels
    
    def generate_confusion_matrix(self, true_labels, predictions, title, save_path=None):
        """Generate and save confusion matrix"""
        unique_labels = sorted(list(set(true_labels + predictions)))
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_roc_curve(self, true_labels, scores, title, save_path=None):
        """Generate and save ROC curve"""
        try:
            fpr, tpr, thresholds = roc_curve(true_labels, scores)
            auc = roc_auc_score(true_labels, scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {title}')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            self.logger.error(f"Error generating ROC curve: {e}")
    
    def save_results(self, output_path='evaluation_results.json'):
        """Save evaluation results to JSON file"""
        # Add timestamp
        self.results['evaluation_timestamp'] = datetime.now().isoformat()
        self.results['api_url'] = self.api_url
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*60)
        print("🎯 FACE RECOGNITION SYSTEM EVALUATION SUMMARY")
        print("="*60)
        
        if 'face_recognition' in self.results:
            fr = self.results['face_recognition']
            print(f"\n📸 FACE RECOGNITION (Your Methodology Metrics):")
            print(f"   Accuracy:  {fr['accuracy']:.4f} ({fr['accuracy']*100:.2f}%)")
            print(f"   Precision: {fr['precision']:.4f} ({fr['precision']*100:.2f}%)")
            print(f"   Recall:    {fr['recall']:.4f} ({fr['recall']*100:.2f}%)")
            print(f"   F1-Score:  {fr['f1_score']:.4f} ({fr['f1_score']*100:.2f}%)")
            print(f"   Total Samples: {fr['total_samples']}")
            print(f"   Correct Predictions: {fr['correct_predictions']}")
            print(f"   Mean Similarity: {fr['mean_similarity']:.4f}")
        
        if 'liveness_detection' in self.results:
            ld = self.results['liveness_detection']
            print(f"\n👁️  LIVENESS DETECTION:")
            print(f"   Accuracy:  {ld['accuracy']:.4f} ({ld['accuracy']*100:.2f}%)")
            print(f"   Precision: {ld['precision']:.4f} ({ld['precision']*100:.2f}%)")
            print(f"   Recall:    {ld['recall']:.4f} ({ld['recall']*100:.2f}%)")
            print(f"   F1-Score:  {ld['f1_score']:.4f} ({ld['f1_score']*100:.2f}%)")
            print(f"   AUC-ROC:   {ld['auc_roc']:.4f}")
            print(f"   Total Samples: {ld['total_samples']} (Live: {ld['live_samples']}, Spoof: {ld['spoof_samples']})")
        
        print(f"\n💡 FOR YOUR THESIS:")
        if 'face_recognition' in self.results:
            fr = self.results['face_recognition']
            print(f"\"The facial recognition system achieved an accuracy of {fr['accuracy']*100:.2f}%, ")
            print(f"precision of {fr['precision']*100:.2f}%, recall of {fr['recall']*100:.2f}%, ")
            print(f"and F1-score of {fr['f1_score']*100:.2f}% when evaluated on {fr['total_samples']} test images.\"")
        
        print("\n" + "="*60)

# Ready-to-use function for your thesis
def run_thesis_evaluation():
    """
    Main function to run evaluation for your thesis
    """
    print("🚀 Starting Face Recognition Evaluation for Thesis")
    print("Using your existing Flask API - no code changes needed!")
    print("="*60)
    
    # Initialize evaluator
    evaluator = FaceRecognitionEvaluator("http://localhost:5000")
    
    # Load test data from directory
    test_images, test_labels = evaluator.load_test_data_from_directory("test_images")
    
    if not test_images:
        print("❌ No test data found!")
        print("Create 'test_images' directory with user subfolders and test images")
        return None
    
    # Run face recognition evaluation
    print(f"\n📸 Evaluating Face Recognition on {len(test_images)} images...")
    fr_results = evaluator.evaluate_face_recognition(test_images, test_labels)
    
    if fr_results:
        # Generate confusion matrix
        print("\n📊 Generating confusion matrix...")
        # Get predictions for confusion matrix
        predictions = []
        for img_path, true_label in zip(test_images, test_labels):
            result = evaluator.call_recognize_api(img_path)
            pred = result.get("user_id", "unknown") if result["status"] == "success" else "unknown"
            predictions.append(pred)
        
        evaluator.generate_confusion_matrix(test_labels, predictions, "Face Recognition", "confusion_matrix.png")
        
        # Print summary
        evaluator.print_summary()
        
        # Save results
        evaluator.save_results('thesis_evaluation_results.json')
        
        print("\n✅ Evaluation completed successfully!")
        print("📁 Results saved to 'thesis_evaluation_results.json'")
        print("📊 Confusion matrix saved to 'confusion_matrix.png'")
        
        return evaluator.results
    else:
        print("❌ Evaluation failed!")
        return None

if __name__ == "__main__":
    # Run the evaluation for your thesis
    results = run_thesis_evaluation()