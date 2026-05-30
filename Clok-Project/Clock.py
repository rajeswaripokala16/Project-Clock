import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(3)

# Capture the background
for i in range(30):
    ret, background = cap.read()

background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range for blue color
    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    # Refining the mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    # Segment out the blue color part
    res1 = cv2.bitwise_and(background, background, mask=mask1)

    # Segment out the rest of the frame
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)

    # Final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Invisible Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
