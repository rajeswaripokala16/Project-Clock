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

    # ---------- CHANGE COLOR HERE ----------
    # Example: Red cloak
    # lower_color = np.array([0, 120, 70])
    # upper_color = np.array([10, 255, 255])
    #q
    # Example: Green cloak
    # lower_color = np.array([35, 40, 40])
    # upper_color = np.array([85, 255, 255])
    #
    # Example: Blue cloak
    lower_color = np.array([100, 40, 40])
    upper_color = np.array([140, 255, 255])
    # ---------------------------------------

    mask1 = cv2.inRange(hsv, lower_color, upper_color)

    # Refining the mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    # Segment out the selected color part
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
