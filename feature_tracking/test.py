import math
import random

import cv2 as cv
import numpy as np
import feature_tracking.tracks

img1 = cv.imread('../data/20240421_154842.png')
img2 = cv.imread('../data/20240421_154905.png')

orb = cv.ORB.create()
bfm = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

prev_kp, prev_des = orb.detectAndCompute(img1, None)
kp, des = orb.detectAndCompute(img2, None)

matches = bfm.match(prev_des, des)

matches = sorted(matches, key=lambda x: x.distance)[:30]

for match in matches:
    prev_pt = prev_kp[match.queryIdx].pt
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    pt = kp[match.trainIdx].pt
    cv.circle(img1, (int(prev_pt[0]), int(prev_pt[1])), 3, color, -1)
    cv.circle(img2, (int(pt[0]), int(pt[1])), 3, color, -1)

m = cv.drawMatches(img1, prev_kp, img2, kp, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow("matches", m)
cv.imshow("img1", img1)
cv.imshow("img2", img2)
cv.waitKey(0)