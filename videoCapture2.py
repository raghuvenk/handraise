import numpy as np
import cv2
from random import randint
from handy import *

raiseHeight = 160
faceWidth = 90
resize = 1

hist = captureHistogram(0)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
capWidth  = int(cap.get(3))
capHeight = int(cap.get(4))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret, frame, countours, defects = detectHand(frame, hist, sketchContours = True, computeDefects = True)
    fingertips = extractFingertips(defects, countours, 50, right = True)

    #saving a face
    if type(countours) == np.ndarray:
        h = countours[0][0][1]
        w = int(countours[0][0][0])
        if h < raiseHeight:
            topRightX = max(w - faceWidth, 0)
            crop_frame = frame[raiseHeight: capHeight - 1, topRightX: w]
            cv2.imwrite("targetFace.jpg", crop_frame)
    #drawing on the frame
            frame = cv2.putText(frame, "Raised!", (topRightX, raiseHeight), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (topRightX, capHeight - 1), (w, raiseHeight + 1), (255, 0, 0), 1)        

    frame = cv2.line(frame, (0, raiseHeight), (int(capWidth - 1), raiseHeight), (255, 255, 255), 1)
    plot(frame, fingertips)
    if resize != 1:
        frame = cv2.resize(frame, (0,0), fx = resize, fy = resize)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
