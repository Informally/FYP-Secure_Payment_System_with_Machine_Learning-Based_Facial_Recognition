import requests
import json
import base64
import cv2
import time
from datetime import datetime

class ManualSpoofingEvaluator:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.spoofing_results = []
    
    def capture_frames_from_webcam(self, duration=3, fps=10):
        """Capture frames from webcam for liveness testing"""
        print(f"📷 Capturing frames for {duration} seconds...")
        print("Look at camera and move your head slightly!")
        
        cap = cv2.VideoCapture(0)
        frames = []
        
        start_time = time.time()
        frame_interval = 1.0 / fps
        last_capture = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if ret and (time.time() - last_capture) >= frame_interval:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_b64)
                last_capture = time.time()
                
                # Show frame
                cv2.imshow('Capturing...', frame)
                cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Captured {len(frames)} frames")
        return frames
    
    def test_legitimate_user(self, user_id):
        """Test legitimate user with live video"""
        print(f"\n🧑 Testing Legitimate User: {user_id}")
        print("Position yourself in front of camera...")
        input("Press Enter when ready...")
        
        frames = self.capture_frames_from_webcam()
        if len(frames) < 6:
            print("❌ Not enough frames captured")
            return None
        
        payload = {
            'image_base64': frames[0],
            'additional_frames': frames[1:],
            'user_id': user_id,
            'challenge_type': 'general',
            'transaction_amount': 100
        }
        
        try:
            response = requests.post(f"{self.api_url}/authenticate", json=payload)
            result = response.json()
            
            test_result = {
                'test_type': 'legitimate_user',
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'status': result.get('status'),
                'liveness_verified': result.get('liveness_verified'),
                'liveness_score': result.get('liveness_score'),
                'similarity': result.get('similarity'),
                'message': result.get('message'),
                'passed': result.get('status') == 'success'
            }
            
            self.spoofing_results.append(test_result)
            
            print(f"Result: {result.get('status')}")
            print(f"Liveness: {result.get('liveness_verified')} (score: {result.get('liveness_score', 0):.3f})")
            print(f"Similarity: {result.get('similarity', 0):.3f}")
            
            return test_result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_photo_spoofing(self, spoof_type="printed_photo"):
        """Test spoofing with static photo"""
        print(f"\n🖼️ Testing {spoof_type.replace('_', ' ').title()}")
        
        if spoof_type == "printed_photo":
            print("📄 Hold a PRINTED PHOTO of a registered user in front of camera")
        elif spoof_type == "phone_screen":
            print("📱 Show a PHONE SCREEN with photo of registered user")
        elif spoof_type == "laptop_screen":
            print("💻 Show a LAPTOP SCREEN with photo of registered user")
        
        input("Press Enter when ready...")
        
        frames = self.capture_frames_from_webcam()
        if len(frames) < 6:
            print("❌ Not enough frames captured")
            return None
        
        payload = {
            'image_base64': frames[0],
            'additional_frames': frames[1:],
            'user_id': None,  # Don't claim specific user
            'challenge_type': 'general',
            'transaction_amount': 100
        }
        
        try:
            response = requests.post(f"{self.api_url}/authenticate", json=payload)
            result = response.json()
            
            test_result = {
                'test_type': 'spoofing_attempt',
                'spoof_method': spoof_type,
                'timestamp': datetime.now().isoformat(),
                'status': result.get('status'),
                'liveness_verified': result.get('liveness_verified'),
                'liveness_score': result.get('liveness_score'),
                'similarity': result.get('similarity'),
                'message': result.get('message'),
                'passed': result.get('status') == 'failed'  # Should FAIL for spoofing
            }
            
            self.spoofing_results.append(test_result)
            
            print(f"Result: {result.get('status')}")
            print(f"Liveness: {result.get('liveness_verified')} (score: {result.get('liveness_score', 0):.3f})")
            
            if result.get('status') == 'failed':
                print("✅ GOOD: Spoofing attempt correctly rejected!")
            else:
                print("❌ BAD: Spoofing attempt was accepted!")
            
            return test_result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_spoofing_evaluation(self, registered_users):
        """Run complete spoofing evaluation"""
        print("🎭 MANUAL SPOOFING EVALUATION")
        print("="*50)
        print("This will test your system's ability to detect spoofing attacks")
        print("Make sure your webcam is working and face API is running!")
        
        # Test 1: Legitimate user baseline
        if registered_users:
            print(f"\n📋 Step 1: Test legitimate user baseline")
            user_to_test = registered_users[0]  # Use first user
            self.test_legitimate_user(user_to_test)
        
        # Test 2: Photo spoofing attempts
        print(f"\n🎭 Step 2: Spoofing attempts")
        print("We'll test different spoofing methods...")
        
        spoof_methods = [
            "printed_photo",
            "phone_screen", 
            "laptop_screen"
        ]
        
        for method in spoof_methods:
            print(f"\n--- Testing {method.replace('_', ' ').title()} ---")
            
            proceed = input(f"Do you have a {method.replace('_', ' ')} ready? (y/n): ")
            if proceed.lower() == 'y':
                self.test_photo_spoofing(method)
            else:
                print(f"Skipping {method}")
        
        # Generate report
        self.generate_spoofing_report()
    
    def generate_spoofing_report(self):
        """Generate spoofing evaluation report"""
        if not self.spoofing_results:
            print("No spoofing test results to report")
            return
        
        print("\n" + "="*50)
        print("🎭 SPOOFING EVALUATION RESULTS")
        print("="*50)
        
        legitimate_tests = [r for r in self.spoofing_results if r['test_type'] == 'legitimate_user']
        spoofing_tests = [r for r in self.spoofing_results if r['test_type'] == 'spoofing_attempt']
        
        # Legitimate user performance
        if legitimate_tests:
            legitimate_passed = sum(1 for t in legitimate_tests if t['passed'])
            legitimate_rate = legitimate_passed / len(legitimate_tests)
            print(f"👤 Legitimate User Recognition: {legitimate_rate:.1%} ({legitimate_passed}/{len(legitimate_tests)})")
        
        # Spoofing detection performance  
        if spoofing_tests:
            spoofing_detected = sum(1 for t in spoofing_tests if t['passed'])
            detection_rate = spoofing_detected / len(spoofing_tests)
            print(f"🛡️ Spoofing Detection Rate: {detection_rate:.1%} ({spoofing_detected}/{len(spoofing_tests)})")
            
            # Breakdown by method
            methods = set(t['spoof_method'] for t in spoofing_tests)
            for method in methods:
                method_tests = [t for t in spoofing_tests if t['spoof_method'] == method]
                method_detected = sum(1 for t in method_tests if t['passed'])
                method_rate = method_detected / len(method_tests) if method_tests else 0
                print(f"   {method.replace('_', ' ').title()}: {method_rate:.1%}")
        
        # Liveness detection analysis
        all_liveness_scores = [t['liveness_score'] for t in self.spoofing_results if t['liveness_score'] is not None]
        if all_liveness_scores:
            avg_liveness = sum(all_liveness_scores) / len(all_liveness_scores)
            print(f"📊 Average Liveness Score: {avg_liveness:.3f}")
        
        # Save detailed results
        with open('spoofing_evaluation_results.json', 'w') as f:
            json.dump({
                'evaluation_date': datetime.now().isoformat(),
                'summary': {
                    'total_tests': len(self.spoofing_results),
                    'legitimate_tests': len(legitimate_tests),
                    'spoofing_tests': len(spoofing_tests),
                    'spoofing_detection_rate': detection_rate if spoofing_tests else 0
                },
                'detailed_results': self.spoofing_results
            }, f, indent=2)
        
        print(f"\n✅ Detailed results saved to 'spoofing_evaluation_results.json'")
        print("="*50)

# Usage example
if __name__ == "__main__":
    # First get registered users from database
    import mysql.connector
    
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root", 
            password="",
            database="payment_facial"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM face_embeddings LIMIT 3")
        registered_users = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        print(f"Found registered users: {registered_users}")
        
        if registered_users:
            evaluator = ManualSpoofingEvaluator()
            evaluator.run_spoofing_evaluation(registered_users)
        else:
            print("No registered users found in database!")
            
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Make sure your face API is running and database is accessible")