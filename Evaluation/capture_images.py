import cv2
import os
import time
from datetime import datetime

class ImageCaptureForTesting:
    def __init__(self):
        self.cap = None
        self.create_directories()
        self.captured_count = {'normal': 0, 'low': 0, 'bright': 0}
        
    def create_directories(self):
        """Create folder structure for test images"""
        base_dir = "test_images"
        subdirs = ["normal_lighting", "low_lighting", "bright_lighting"]
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        for subdir in subdirs:
            full_path = os.path.join(base_dir, subdir)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                print(f"Created directory: {full_path}")
    
    def initialize_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully!")
        return True
    
    def capture_images_interactive(self):
        """Interactive image capture with lighting condition selection"""
        if not self.initialize_camera():
            return
        
        print("\n=== Face Detection Test Image Capture ===")
        print("Instructions:")
        print("- Position your face in the center of the frame")
        print("- Press SPACE to capture image")
        print("- Press 'q' to quit current mode")
        print("- Press ESC to exit completely")
        print("\nLighting modes:")
        print("1 = Normal lighting")
        print("2 = Low lighting") 
        print("3 = Bright lighting")
        
        current_mode = None
        mode_names = {1: 'normal', 2: 'low', 3: 'bright'}
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add text overlay
            if current_mode:
                mode_text = f"Mode: {mode_names[current_mode].upper()} LIGHTING"
                count_text = f"Captured: {self.captured_count[mode_names[current_mode]]}"
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Capture | Q: Change mode | ESC: Exit", (10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Press 1, 2, or 3 to select lighting mode", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "1=Normal | 2=Low Light | 3=Bright Light", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw face detection guide
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            cv2.rectangle(frame, (center_x-100, center_y-120), (center_x+100, center_y+120), (255, 0, 0), 2)
            cv2.putText(frame, "Position face here", (center_x-90, center_y-130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            cv2.imshow('Face Detection Test - Image Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # ESC to exit completely
            if key == 27:
                break
            
            # Select lighting mode
            elif key in [ord('1'), ord('2'), ord('3')]:
                current_mode = int(chr(key))
                print(f"\nSwitched to {mode_names[current_mode].upper()} LIGHTING mode")
                print("Adjust your lighting now, then press SPACE to capture images")
                
                if current_mode == 2:
                    print("💡 TIP: Turn off lights or move to darker area")
                elif current_mode == 3:
                    print("💡 TIP: Turn on bright lights or move near window")
                else:
                    print("💡 TIP: Use normal room lighting")
            
            # Capture image
            elif key == ord(' ') and current_mode:
                success = self.save_image(frame, mode_names[current_mode])
                if success:
                    print(f"✅ Captured image in {mode_names[current_mode]} lighting mode")
                    # Brief flash effect
                    flash_frame = frame.copy()
                    flash_frame[:] = (255, 255, 255)
                    cv2.imshow('Face Detection Test - Image Capture', flash_frame)
                    cv2.waitKey(100)
            
            # Quit current mode
            elif key == ord('q'):
                current_mode = None
                print("\nReturned to mode selection")
        
        self.cleanup()
    
    def save_image(self, frame, lighting_mode):
        """Save captured image with proper naming"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            count = self.captured_count[lighting_mode] + 1
            
            filename = f"{lighting_mode}_lighting_{count:02d}_{timestamp}.jpg"
            filepath = os.path.join("test_images", f"{lighting_mode}_lighting", filename)
            
            # Save image
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self.captured_count[lighting_mode] += 1
                print(f"Saved: {filepath}")
                return True
            else:
                print(f"Failed to save image: {filepath}")
                return False
                
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def auto_capture_session(self, images_per_condition=5, delay_between_captures=3):
        """Automated capture session with prompts"""
        if not self.initialize_camera():
            return
        
        conditions = [
            ("normal_lighting", "NORMAL LIGHTING", "Set up normal room lighting"),
            ("low_lighting", "LOW LIGHTING", "Turn off lights or move to darker area"),
            ("bright_lighting", "BRIGHT LIGHTING", "Turn on bright lights or move near window")
        ]
        
        print("\n=== Automated Capture Session ===")
        print(f"Will capture {images_per_condition} images per lighting condition")
        print(f"Delay between captures: {delay_between_captures} seconds")
        
        for condition_folder, condition_name, instruction in conditions:
            print(f"\n📸 {condition_name} SESSION")
            print(f"🔧 {instruction}")
            input("Press ENTER when lighting is ready...")
            
            print("Position yourself in front of the camera...")
            
            # Show preview for 3 seconds
            for i in range(30):
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"GET READY - {condition_name}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Starting in {3 - i//10} seconds", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Face guide
                    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                    cv2.rectangle(frame, (center_x-100, center_y-120), (center_x+100, center_y+120), (255, 0, 0), 2)
                    
                    cv2.imshow('Auto Capture Session', frame)
                    cv2.waitKey(100)
            
            # Capture images
            for img_num in range(1, images_per_condition + 1):
                print(f"Capturing image {img_num}/{images_per_condition}...")
                
                # Countdown
                for countdown in range(delay_between_captures, 0, -1):
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"{condition_name} - Image {img_num}/{images_per_condition}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Capturing in {countdown}...", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Face guide
                        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                        cv2.rectangle(frame, (center_x-100, center_y-120), (center_x+100, center_y+120), (255, 0, 0), 2)
                        
                        cv2.imshow('Auto Capture Session', frame)
                        cv2.waitKey(1000)
                
                # Capture the image
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    # Show capture moment
                    cv2.putText(frame, "CAPTURED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.imshow('Auto Capture Session', frame)
                    cv2.waitKey(500)
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{condition_folder.split('_')[0]}_{img_num:02d}_{timestamp}.jpg"
                    filepath = os.path.join("test_images", condition_folder, filename)
                    
                    cv2.imwrite(filepath, frame)
                    print(f"✅ Saved: {filename}")
                
                time.sleep(0.5)  # Brief pause between captures
        
        print("\n🎉 All images captured successfully!")
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n=== Capture Summary ===")
        total = sum(self.captured_count.values())
        print(f"Total images captured: {total}")
        for condition, count in self.captured_count.items():
            print(f"  {condition.capitalize()} lighting: {count} images")
        print(f"\nImages saved in: test_images/")

def main():
    """Main function with menu options"""
    print("=== Face Detection Test Image Capture Tool ===")
    print("Choose capture mode:")
    print("1. Interactive Mode (manual control)")
    print("2. Auto Mode (automated capture)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            capturer = ImageCaptureForTesting()
            capturer.capture_images_interactive()
            break
        elif choice == '2':
            images_per_condition = int(input("Images per lighting condition (default 5): ") or "5")
            capturer = ImageCaptureForTesting()
            capturer.auto_capture_session(images_per_condition=images_per_condition)
            break
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()