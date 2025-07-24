import cv2
import os
import time
from datetime import datetime
import zipfile

class FriendFaceCapture:
    def __init__(self):
        self.create_output_folder()
        
    def create_output_folder(self):
        """Create output folder for captured images"""
        self.output_folder = "my_face_images"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created folder: {self.output_folder}")
    
    def capture_my_images(self, my_name, num_images=8):
        """Capture images for the person using this tool"""
        print(f"\n📸 Hi {my_name}! Let's capture some photos for face recognition")
        print(f"I'll take {num_images} photos of you")
        print("\n📋 INSTRUCTIONS:")
        print("✅ Look directly at the camera")
        print("✅ Make sure your face is well-lit")
        print("✅ Try different slight angles (left, right, straight)")
        print("✅ Try different expressions (neutral, smile)")
        print("✅ Keep your face in the blue rectangle")
        print("\n⌨️  CONTROLS:")
        print("   SPACEBAR = Take photo")
        print("   Q = Finish early")
        print("   ESC = Cancel")
        
        input("\nPress ENTER when you're ready to start...")
        
        # Initialize webcam - try different camera indices
        cap = None
        for camera_index in [0, 1, 2]:
            print(f"Trying camera {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Test if camera actually works
                ret, test_frame = cap.read()
                if ret:
                    print(f"✅ Camera {camera_index} is working!")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            print("❌ Error: Could not open any webcam")
            print("💡 Troubleshooting:")
            print("   1. Close all apps using camera (Zoom, Skype, etc.)")
            print("   2. Unplug and replug your webcam")
            print("   3. Try restarting your computer")
            print("   4. Check if camera works in other apps")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        captured_count = 0
        
        print(f"\n🎬 Camera started! Position your face in the blue rectangle")
        
        while captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
            
            # Flip frame for mirror effect (more natural)
            frame = cv2.flip(frame, 1)
            
            # Add informative overlay
            cv2.putText(frame, f"Capturing for: {my_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Photos taken: {captured_count}/{num_images}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(frame, "SPACEBAR: Take Photo | Q: Finish | ESC: Cancel", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Face positioning guide
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            cv2.rectangle(frame, (center_x-120, center_y-140), (center_x+120, center_y+140), (255, 0, 0), 3)
            cv2.putText(frame, "Keep face in this box", (center_x-110, center_y-150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Tips based on progress
            if captured_count == 0:
                tip = "Look straight at camera"
            elif captured_count == 1:
                tip = "Try turning slightly left"
            elif captured_count == 2:
                tip = "Try turning slightly right"
            elif captured_count == 3:
                tip = "Try smiling"
            elif captured_count == 4:
                tip = "Back to neutral expression"
            else:
                tip = "Different angles are good!"
            
            cv2.putText(frame, f"Tip: {tip}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(f'Face Capture - {my_name}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # ESC to cancel
            if key == 27:
                print("\n❌ Capture cancelled")
                break
            
            # 'q' to finish early
            elif key == ord('q'):
                if captured_count >= 3:
                    print(f"\n✅ Finished early with {captured_count} photos (minimum reached)")
                    break
                else:
                    print(f"\n⚠️  Need at least 3 photos. Currently have {captured_count}")
            
            # SPACE to capture
            elif key == ord(' '):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{my_name.replace(' ', '_')}_photo_{captured_count+1:02d}_{timestamp}.jpg"
                filepath = os.path.join(self.output_folder, filename)
                
                # Save image
                success = cv2.imwrite(filepath, frame)
                
                if success:
                    captured_count += 1
                    print(f"✅ Photo {captured_count} saved: {filename}")
                    
                    # Success flash effect
                    flash_frame = frame.copy()
                    cv2.rectangle(flash_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 20)
                    cv2.putText(flash_frame, f"PHOTO {captured_count} SAVED!", (center_x-150, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.imshow(f'Face Capture - {my_name}', flash_frame)
                    cv2.waitKey(500)
                    
                    if captured_count < num_images:
                        print(f"👍 Great! Now let's take photo {captured_count + 1}")
                else:
                    print(f"❌ Failed to save photo. Please try again.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_count >= 3:
            print(f"\n🎉 Capture completed successfully!")
            print(f"📊 Total photos captured: {captured_count}")
            return True
        else:
            print(f"\n❌ Not enough photos captured. Need at least 3, got {captured_count}")
            return False
    
    def create_zip_package(self, person_name):
        """Create a zip file for easy sharing"""
        if not os.path.exists(self.output_folder):
            print("❌ No images to package")
            return None
        
        image_files = [f for f in os.listdir(self.output_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("❌ No image files found")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{person_name.replace(' ', '_')}_face_photos_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for image_file in image_files:
                file_path = os.path.join(self.output_folder, image_file)
                zipf.write(file_path, image_file)
        
        print(f"\n📦 Created zip package: {zip_filename}")
        print(f"📁 Contains {len(image_files)} photos")
        print(f"📤 Send this zip file to your friend!")
        
        return zip_filename
    
    def show_summary(self, person_name):
        """Show summary of captured images"""
        if not os.path.exists(self.output_folder):
            return
        
        image_files = [f for f in os.listdir(self.output_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📋 CAPTURE SUMMARY FOR {person_name.upper()}")
        print("=" * 50)
        print(f"📸 Total photos: {len(image_files)}")
        print(f"📁 Saved in folder: {self.output_folder}")
        print(f"📝 Files:")
        
        for i, filename in enumerate(sorted(image_files), 1):
            file_path = os.path.join(self.output_folder, filename)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   {i}. {filename} ({file_size:.1f} KB)")

def main():
    """Main function - simple and user-friendly"""
    
    print("=" * 60)
    print("🤖 FACE RECOGNITION PHOTO CAPTURE TOOL")
    print("=" * 60)
    print("👋 Hi! This tool will help capture photos of your face")
    print("🎯 Your friend needs these photos for their face recognition project")
    print("⏱️  This will take about 2-3 minutes")
    print("📷 Make sure you have good lighting and a working webcam")
    
    print("\n📋 REQUIREMENTS:")
    print("✅ Working webcam")
    print("✅ Good lighting (not too dark)")
    print("✅ Stable internet (if sending files)")
    print("✅ About 3 minutes of time")
    
    # Get person's name
    while True:
        person_name = input("\n👤 What's your name? ").strip()
        if person_name:
            break
        else:
            print("❌ Please enter your name")
    
    print(f"\n👋 Hi {person_name}! Let's get started.")
    
    # Ask about number of photos
    print(f"\n📸 How many photos should we take?")
    print("   Recommended: 8 photos (takes ~2 minutes)")
    print("   Minimum: 5 photos")
    print("   Maximum: 12 photos")
    
    while True:
        try:
            num_photos = input("\nEnter number of photos (5-12, default 8): ").strip()
            if not num_photos:
                num_photos = 8
                break
            num_photos = int(num_photos)
            if 5 <= num_photos <= 12:
                break
            else:
                print("❌ Please enter a number between 5 and 12")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"\n🎯 Perfect! We'll capture {num_photos} photos of you")
    
    # Start capture process
    capturer = FriendFaceCapture()
    
    success = capturer.capture_my_images(person_name, num_photos)
    
    if success:
        # Show summary
        capturer.show_summary(person_name)
        
        # Ask about creating zip
        print(f"\n📦 Would you like to create a zip file for easy sharing?")
        create_zip = input("Create zip file? (y/n, default yes): ").strip().lower()
        
        if create_zip in ['', 'y', 'yes']:
            zip_file = capturer.create_zip_package(person_name)
            if zip_file:
                print(f"\n✅ SUCCESS! Everything is ready")
                print(f"📤 Send this file to your friend: {zip_file}")
                print(f"💡 You can also send the entire '{capturer.output_folder}' folder")
        
        print(f"\n🎉 All done! Thank you {person_name}!")
        print("👋 You can close this program now")
        
    else:
        print(f"\n❌ Capture failed. Please try again.")
        print("💡 Make sure your webcam is working and try again")
    
    input("\nPress ENTER to exit...")

if __name__ == "__main__":
    main()