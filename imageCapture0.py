import numpy as np
import cv2
from random import randint
from handy import *

frame = cv2.imread("9 elementary colors.png")
print(frame)
cv2.imshow('image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
