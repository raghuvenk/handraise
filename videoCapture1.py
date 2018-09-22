import numpy as np
import cv2
from random import randint
from handy import *

hist = captureHistogram(0)
print(hist)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret, frame, countours, defects = detectHand(frame, hist, sketchContours = True, computeDefects = True)
    fingertips = extractFingertips(defects, countours, 50, right = True)
    plot(frame, fingertips)
    
    #frame = cv2.resize(frame, (0,0), fx = 2, fy = 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
