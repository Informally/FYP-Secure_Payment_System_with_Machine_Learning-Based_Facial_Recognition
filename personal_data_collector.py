import cv2
import os
import time
import numpy as np
from datetime import datetime

class PersonalLivenessCollector:
    def __init__(self, output_dir='PersonalLivenessData'):
        self.output_dir = output_dir
        self.real_dir = os.path.join(output_dir, 'real')
        self.fake_dir = os.path.join(output_dir, 'fake')
        
        # Create directories
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Data will be saved to: {self.output_dir}")
        
    def apply_lighting_variation(self, frame, variation_type):
        """Apply different lighting conditions"""
        frame_float = frame.astype(np.float32) / 255.0
        
        if variation_type == 'very_dim':
            frame_float = frame_float * np.random.uniform(0.2, 0.4)
            
        elif variation_type == 'dim':
            frame_float = frame_float * np.random.uniform(0.4, 0.7)
            
        elif variation_type == 'bright':
            frame_float = np.clip(frame_float * np.random.uniform(1.2, 1.8), 0, 1)
            
        elif variation_type == 'screen_glow':
            # Simulate blue screen glow
            h, w = frame_float.shape[:2]
            center_x, center_y = w//2, h//2
            y, x = np.ogrid[:h, :w]
            mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) * 0.4)**2))
            
            blue_intensity = np.random.uniform(0.2, 0.5)
            frame_float[:,:,0] += mask * blue_intensity * 0.3  # Red
            frame_float[:,:,1] += mask * blue_intensity * 0.5  # Green  
            frame_float[:,:,2] += mask * blue_intensity * 0.8  # Blue
            
        elif variation_type == 'uneven':
            # Uneven lighting
            h, w = frame_float.shape[:2]
            gradient = np.linspace(0.3, 1.2, w)
            for y in range(h):
                frame_float[y, :, :] *= gradient[:, np.newaxis]
        
        frame_float = np.clip(frame_float, 0, 1)
        return (frame_float * 255).astype(np.uint8)
    
    def collect_real_data(self, num_base_samples=30):
        """Collect your real face data under various conditions"""
        
        print(f"\n=== Collecting YOUR REAL face data ===")
        print("Instructions for HIGH-QUALITY real face data:")
        print("- Look directly at camera")
        print("- Move head slightly (left, right, up, down)")
        print("- Try different expressions (smile, neutral, serious)")
        print("- Blink naturally")
        print("- Move closer/farther from camera")
        print("- SPACE = Capture image, ESC = Finish")
        print("\nWe'll create multiple lighting variations automatically!")
        
        lighting_variations = ['normal', 'dim', 'very_dim', 'bright', 'screen_glow', 'uneven']
        captured = 0
        
        while captured < num_base_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Show preview
            display_frame = frame.copy()
            cv2.putText(display_frame, f"REAL Face Collection: {captured}/{num_base_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE=Capture YOUR real face, ESC=Finish", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Move head, change expressions, blink naturally", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            cv2.imshow('Personal Real Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                timestamp = int(time.time() * 1000)
                
                # Save original
                base_filename = f"real_{self.session_id}_{timestamp}"
                cv2.imwrite(os.path.join(self.real_dir, f"{base_filename}_normal.jpg"), frame)
                
                # Create lighting variations
                variation_count = 0
                for lighting in lighting_variations[1:]:  # Skip 'normal'
                    try:
                        modified_frame = self.apply_lighting_variation(frame, lighting)
                        var_filename = f"{base_filename}_{lighting}.jpg"
                        cv2.imwrite(os.path.join(self.real_dir, var_filename), modified_frame)
                        variation_count += 1
                    except Exception as e:
                        print(f"Error creating {lighting} variation: {e}")
                
                captured += 1
                total_images = 1 + variation_count  # Original + variations
                print(f"Captured real sample {captured}: 1 original + {variation_count} variations = {total_images} images")
                time.sleep(0.5)
                
            elif key == 27:  # ESC
                break
        
        total_real_images = len([f for f in os.listdir(self.real_dir) if f.endswith('.jpg')])
        print(f"Real data collection completed: {total_real_images} total images from {captured} base captures")
    
    def collect_fake_data(self, num_samples=20):
        """Collect spoofed data (photos of you on screens/prints)"""
        
        print(f"\n=== Collecting FAKE/SPOOF data ===")
        print("Instructions for spoof data:")
        print("- Take photos of yourself with your phone")
        print("- Show these photos to the camera")
        print("- Use different devices (phone, tablet, laptop screen)")
        print("- Try printed photos")
        print("- Vary angles and distances")
        print("- SPACE = Capture spoof, ESC = Finish")
        
        captured = 0
        
        while captured < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"FAKE/SPOOF Collection: {captured}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, "Show photos/screens of YOUR face to camera", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Use phone, tablet, printed photos", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(display_frame, "SPACE=Capture, ESC=Finish", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('Personal Fake Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                timestamp = int(time.time() * 1000)
                filename = f"fake_{self.session_id}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(self.fake_dir, filename), frame)
                captured += 1
                print(f"Captured fake sample {captured}: {filename}")
                time.sleep(0.3)
                
            elif key == 27:  # ESC
                break
        
        print(f"Fake data collection completed: {captured} samples")
    
    def quick_video_capture(self, duration_seconds=10, fps=2):
        """Capture a video of yourself for extracting multiple frames"""
        
        print(f"\n=== Quick Video Capture (Real Faces) ===")
        print(f"Recording {duration_seconds} seconds at {fps} FPS")
        print("Instructions:")
        print("- Look at camera and move naturally")
        print("- Turn head left, right, up, down")
        print("- Blink, smile, change expressions")
        print("- Move closer and farther")
        print("Press SPACE to start recording...")
        
        # Wait for space to start
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return
                
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press SPACE to start video recording", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Will record {duration_seconds}s at {fps} FPS", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Video Capture Preparation', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        
        # Start recording
        print("Recording started! Move naturally...")
        start_time = time.time()
        frame_interval = 1.0 / fps
        last_save_time = 0
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time() - start_time
            
            # Save frame at specified FPS
            if current_time - last_save_time >= frame_interval:
                timestamp = int(time.time() * 1000)
                base_filename = f"video_real_{self.session_id}_{timestamp}"
                
                # Save original frame
                cv2.imwrite(os.path.join(self.real_dir, f"{base_filename}_normal.jpg"), frame)
                
                # Create lighting variations
                lighting_variations = ['dim', 'very_dim', 'bright', 'screen_glow']
                for lighting in lighting_variations:
                    try:
                        modified_frame = self.apply_lighting_variation(frame, lighting)
                        var_filename = f"{base_filename}_{lighting}.jpg"
                        cv2.imwrite(os.path.join(self.real_dir, var_filename), modified_frame)
                    except Exception as e:
                        print(f"Error creating variation: {e}")
                
                frame_count += 1
                last_save_time = current_time
            
            # Show recording progress
            display_frame = frame.copy()
            remaining = duration_seconds - current_time
            cv2.putText(display_frame, f"RECORDING: {remaining:.1f}s remaining", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Frames captured: {frame_count}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Move head, blink, change expressions!", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Video Recording', display_frame)
            cv2.waitKey(1)
        
        total_images = frame_count * 5  # Original + 4 variations per frame
        print(f"Video capture completed! Captured {frame_count} frames = {total_images} total images")
    
    def run_collection_session(self):
        """Run the complete personal data collection session"""
        
        print("=== PERSONAL LIVENESS DATA COLLECTION ===")
        print("This will help improve your liveness model specifically for YOU!")
        print(f"Session ID: {self.session_id}")
        print(f"Data will be saved to: {self.output_dir}")
        
        while True:
            print("\n" + "="*50)
            print("Collection Options:")
            print("1. Collect real face data (manual capture)")
            print("2. Quick video capture (automatic)")
            print("3. Collect fake/spoof data")
            print("4. Check current data counts")
            print("5. Exit and start training")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                num_samples = int(input("Number of base captures (will create 6x variations each): ") or "20")
                self.collect_real_data(num_samples)
                
            elif choice == '2':
                duration = int(input("Video duration in seconds: ") or "15")
                fps = int(input("Frames per second to extract: ") or "2")
                self.quick_video_capture(duration, fps)
                
            elif choice == '3':
                num_samples = int(input("Number of fake samples: ") or "15")
                self.collect_fake_data(num_samples)
                
            elif choice == '4':
                real_count = len([f for f in os.listdir(self.real_dir) if f.endswith('.jpg')])
                fake_count = len([f for f in os.listdir(self.fake_dir) if f.endswith('.jpg')])
                print(f"\nCurrent data counts:")
                print(f"Real images: {real_count}")
                print(f"Fake images: {fake_count}")
                print(f"Total: {real_count + fake_count}")
                
                if real_count >= 50 and fake_count >= 10:
                    print("✅ You have enough data to start training!")
                elif real_count >= 20:
                    print("✅ Good amount of real data. Consider adding more fake samples.")
                else:
                    print("⚠️  Consider collecting more real face data for better results.")
                
            elif choice == '5':
                break
            
            else:
                print("Invalid choice")
        
        self.cleanup()
        
        # Final summary
        real_count = len([f for f in os.listdir(self.real_dir) if f.endswith('.jpg')])
        fake_count = len([f for f in os.listdir(self.fake_dir) if f.endswith('.jpg')])
        
        print(f"\n=== COLLECTION COMPLETE ===")
        print(f"Final counts:")
        print(f"Real images: {real_count}")
        print(f"Fake images: {fake_count}")
        print(f"Data saved to: {self.output_dir}")
        
        if real_count >= 30:
            print("\n🎉 Excellent! You have enough data for training.")
            print("Next steps:")
            print("1. Run the combined training script")
            print("2. This will train a model specifically optimized for YOUR face")
            print("3. Replace your API model with the new trained model")
        else:
            print("\n⚠️  You might want to collect more real face data for better results.")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera resources released")

def main():
    print("Personal Liveness Data Collector")
    print("This will create training data specifically for YOUR face!")
    
    # Ask for output directory
    output_dir = input("Enter output directory (default: PersonalLivenessData): ").strip()
    if not output_dir:
        output_dir = "PersonalLivenessData"
    
    collector = PersonalLivenessCollector(output_dir)
    collector.run_collection_session()
    
    print("\nData collection finished!")
    print("You can now run the combined training script to create a personalized model.")

if __name__ == "__main__":
    main()