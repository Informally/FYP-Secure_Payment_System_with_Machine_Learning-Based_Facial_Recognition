import cv2
import os

# Create directory if it doesn't exist
os.makedirs('dataset/real', exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 0
print("Capturing real face images... Press 'q' to stop after capturing enough images.")

while count < 200:  # Capture 200 real images (adjust if needed)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow('Capture Real Faces', frame)

    # Save the frame as an image
    cv2.imwrite(f'dataset/real/real_{count}.jpg', frame)
    print(f"Saved real_{count}.jpg")
    count += 1

    # Press 'q' to stop early
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} real images.")