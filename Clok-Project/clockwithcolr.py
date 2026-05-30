import cv2
import numpy as np
import time

# Define HSV color ranges for multiple cloth colors (adjust as needed)
color_ranges = [
    # Green
    (np.array([50, 80, 50]), np.array([90, 255, 255])),
    # Blue
    (np.array([100, 150, 0]), np.array([140, 255, 255])),
    # Red - 2 ranges for red hue wrap-around
    (np.array([0, 120, 70]), np.array([10, 255, 255])),
    (np.array([170, 120, 70]), np.array([180, 255, 255])),
    # Yellow
    (np.array([20, 100, 100]), np.array([30, 255, 255])),
]

# Initialize webcam
cap = cv2.VideoCapture(0)
time.sleep(2)

# Capture background for cloak effect
background = None
for i in range(60):
    ret, background = cap.read()
    if not ret:
        continue
    background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = np.flip(frame, axis=1)  # Mirror the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Initialize empty mask
    final_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    # Combine masks for all defined colors
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        final_mask = cv2.bitwise_or(final_mask, mask)
    
    # Morphological operations to reduce noise
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    final_mask = cv2.dilate(final_mask, np.ones((3,3), np.uint8), iterations=1)
    
    # Invert mask to segment out cloak
    mask_inv = cv2.bitwise_not(final_mask)
    
    # Background part seen through cloak
    cloak_area = cv2.bitwise_and(background, background, mask=final_mask)
    
    # Non-cloak parts
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Final frame
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)
    
    cv2.imshow('Invisible Cloak', final_output)
    
    # Key press handling - close with 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Closing program")
        break

cap.release()              # Release camera resource
cv2.destroyAllWindows()    # Close OpenCV windows
