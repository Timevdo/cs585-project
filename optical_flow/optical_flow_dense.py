# Code from https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
# Intended to test viability of optical flow helping segment the road, the shield, and the inside of the vehicle

import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../data/audi_raw_data.mp4")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

frame_num = 0

while(1):
    ret, frame2 = cap.read()
    frame_num += 1
    if not ret:
        print('No frames grabbed!')
        break

    if frame_num < 4000:
        continue

    original = frame2.copy()

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    cv.imshow("original", original)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb2.png', frame2)
        cv.imwrite('opticalhsv2.png', bgr)
    prvs = next

cv.destroyAllWindows()
