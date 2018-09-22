import numpy as np
import cv2
from random import randint
from handy import *

hist = captureHistogram(0)
print(hist)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
raiseHeight = 200
capWidth = cap.get(3)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret, frame, countours, defects = detectHand(frame, hist, sketchContours = True, computeDefects = True)
    fingertips = extractFingertips(defects, countours, 50, right = True)
    plot(frame, fingertips)

    #drawing a line above which the hand counts as raised
    frame = cv2.line(frame, (0, 200), (int(capWidth - 1), 200), (255, 255, 0), 1)
    if type(countours) == np.ndarray
        m = countours.mean()
        if countours.mean() < raiseHeight:
            cv2.putText(frame, "Raised!", (50, 350), font, 4, (0, 0, 255), 2, cv2.LINE_AA)
    
    #frame = cv2.resize(frame, (0,0), fx = 2, fy = 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
