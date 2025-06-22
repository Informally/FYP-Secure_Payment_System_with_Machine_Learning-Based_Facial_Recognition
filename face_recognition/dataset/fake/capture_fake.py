import cv2
import os

# Create directory if it doesn't exist
os.makedirs('dataset/fake', exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 0
print("Capturing fake (replay attack) images... Hold your phone showing a video of your face in front of the webcam. Press 'q' to stop.")

while count < 200:  # Capture 200 fake images (adjust if needed)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow('Capture Fake Faces (Phone Replay)', frame)

    # Save the frame as an image
    cv2.imwrite(f'dataset/fake/fake_{count}.jpg', frame)
    print(f"Saved fake_{count}.jpg")
    count += 1

    # Press 'q' to stop early
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} fake images.")