# simple_webcam_collector.py
# Complete webcam collector for your test images

import cv2
import os
import time
from datetime import datetime

def collect_test_images():
    """Collect test images for your user ID using webcam"""
    
    # Your user ID
    user_id = "USER_685eb862519131.70324980"
    
    # Create directory
    base_dir = "test_images"
    user_dir = os.path.join(base_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    print("📸 WEBCAM TEST IMAGE COLLECTOR")
    print("=" * 40)
    print(f"👤 User: {user_id}")
    print("🎯 Goal: Collect 5-8 test images")
    print()
    print("💡 Tips for good test images:")
    print("   - Try different lighting")
    print("   - Different expressions (smile, neutral)")
    print("   - Slightly different angles")
    print("   - Different distances from camera")
    print()
    print("🎮 Controls:")
    print("   SPACE: Capture image")
    print("   ESC: Finish")
    print("   Q: Quit")
    print("=" * 40)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("Make sure no other app is using the camera")
        return False
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    collected = 0
    target_images = 6  # Collect 6 test images
    
    print(f"📹 Webcam started! Position yourself and press SPACE to capture")
    print()
    
    while collected < target_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Create display frame
        display_frame = frame.copy()
        
        # Add text overlay
        cv2.putText(display_frame, f"Test Images: {collected}/{target_images}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(display_frame, "SPACE: Capture | ESC: Finish | Q: Quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add face frame guide
        height, width = display_frame.shape[:2]
        face_size = 200
        start_x = (width - face_size) // 2
        start_y = (height - face_size) // 2
        end_x = start_x + face_size
        end_y = start_y + face_size
        
        cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(display_frame, "Position face here", (start_x, start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show tips based on collected images
        if collected == 0:
            tip = "Tip: Normal expression, good lighting"
        elif collected == 1:
            tip = "Tip: Try smiling"
        elif collected == 2:
            tip = "Tip: Turn head slightly left"
        elif collected == 3:
            tip = "Tip: Turn head slightly right"
        elif collected == 4:
            tip = "Tip: Move closer to camera"
        else:
            tip = "Tip: Move further from camera"
        
        cv2.putText(display_frame, tip, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('Test Image Collector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_{collected+1}_{timestamp}.jpg"
            filepath = os.path.join(user_dir, filename)
            
            # Save the original frame (without overlay)
            cv2.imwrite(filepath, frame)
            collected += 1
            
            print(f"✅ Captured: {filename}")
            
            # Show confirmation on screen
            conf_frame = display_frame.copy()
            cv2.putText(conf_frame, f"CAPTURED! Image {collected}", 
                       (width//2 - 100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow('Test Image Collector', conf_frame)
            cv2.waitKey(500)  # Show for 500ms
            
        elif key == 27:  # ESC to finish
            print(f"⏹️  Finished early with {collected} images")
            break
            
        elif key == ord('q'):  # Q to quit
            print("❌ Quit by user")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("🎉 COLLECTION COMPLETED!")
    print("=" * 30)
    print(f"✅ Collected: {collected} test images")
    print(f"📁 Saved to: {user_dir}")
    print()
    
    # List the captured files
    if os.path.exists(user_dir):
        images = [f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')]
        print("📋 Captured files:")
        for img in images:
            print(f"   📸 {img}")
    
    print()
    print("🚀 Next steps:")
    print(f"1. Check images in folder: {user_dir}")
    print("2. Collect images for other users (optional)")
    print("3. Run evaluation: python face_recognition_evaluator.py")
    
    return True

def quick_test_single_user():
    """Quick test with just your images"""
    print("🧪 QUICK SINGLE-USER TEST")
    print("=" * 30)
    
    user_id = "USER_685eb862519131.70324980"
    user_dir = os.path.join("test_images", user_id)
    
    if not os.path.exists(user_dir):
        print("❌ No test images found!")
        print("Run collect_test_images() first")
        return
    
    images = [f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')]
    
    if len(images) < 3:
        print(f"❌ Need at least 3 test images, found {len(images)}")
        print("Collect more images first")
        return
    
    print(f"✅ Found {len(images)} test images")
    print("🚀 Ready for evaluation!")
    print()
    print("Run this command:")
    print("python face_recognition_evaluator.py")

def check_status():
    """Check current test image status"""
    print("📊 CURRENT STATUS")
    print("=" * 20)
    
    user_id = "USER_685eb862519131.70324980"
    user_dir = os.path.join("test_images", user_id)
    
    if not os.path.exists(user_dir):
        print("❌ No test_images directory found")
        print("💡 Run option 1 to start collecting")
        return
    
    images = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"👤 User: {user_id}")
    print(f"📁 Directory: {user_dir}")
    print(f"📸 Images found: {len(images)}")
    
    if images:
        print("📋 Files:")
        for img in images:
            print(f"   • {img}")
    
    if len(images) >= 3:
        print("✅ Ready for evaluation!")
    else:
        print(f"⏳ Need {3 - len(images)} more images")

def main():
    """Main function"""
    print("🎯 WEBCAM TEST IMAGE COLLECTOR")
    print("=" * 40)
    print("Choose an option:")
    print("1. Collect test images with webcam")
    print("2. Check current status")
    print("3. Test single user evaluation readiness")
    
    choice = input("\nEnter choice (1/2/3): ")
    
    if choice == '1':
        success = collect_test_images()
        if success:
            print("\n" + "="*40)
            quick_test_single_user()
    elif choice == '2':
        check_status()
    elif choice == '3':
        quick_test_single_user()
    else:
        print("❌ Invalid choice")
        print("Please run again and choose 1, 2, or 3")

if __name__ == "__main__":
    main()