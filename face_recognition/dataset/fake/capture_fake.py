import cv2
import os
import glob

# Create directory if it doesn't exist
os.makedirs('dataset/fake', exist_ok=True)

# Find existing fake images and get the next count number
existing_files = glob.glob('dataset/fake/fake_*.jpg')
if existing_files:
    # Extract numbers from existing filenames and find the maximum
    existing_numbers = []
    for file in existing_files:
        try:
            # Extract number from filename like 'fake_123.jpg'
            filename = os.path.basename(file)
            number = int(filename.split('_')[1].split('.')[0])
            existing_numbers.append(number)
        except (ValueError, IndexError):
            continue
    
    # Start counting from the next number after the highest existing number
    start_count = max(existing_numbers) + 1 if existing_numbers else 0
    print(f"Found {len(existing_files)} existing fake images. Starting from fake_{start_count}.jpg")
else:
    start_count = 0
    print("No existing fake images found. Starting from fake_0.jpg")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = start_count
target_count = start_count + 200  # Capture 200 new fake images
print(f"Capturing fake (replay attack) images... Hold your phone showing a video of your face in front of the webcam.")
print(f"Will capture from fake_{start_count}.jpg to fake_{target_count-1}.jpg")
print("Press 'q' to stop.")

while count < target_count:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow('Capture Fake Faces (Phone Replay)', frame)

    # Save the frame as an image
    filename = f'dataset/fake/fake_{count}.jpg'
    cv2.imwrite(filename, frame)
    print(f"Saved fake_{count}.jpg ({count - start_count + 1}/{target_count - start_count})")
    count += 1

    # Press 'q' to stop early
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final summary
total_captured = count - start_count
total_fake_images = len(glob.glob('dataset/fake/fake_*.jpg'))
print(f"\nCapture complete!")
print(f"Newly captured: {total_captured} fake images")
print(f"Total fake images in dataset: {total_fake_images}")
print(f"Range of new images: fake_{start_count}.jpg to fake_{count-1}.jpg")