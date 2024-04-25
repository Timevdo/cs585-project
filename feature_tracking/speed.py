import math
import random

import cv2
import numpy as np
import pytesseract

from feature_tracking.tracks import *

def extract_speed_from_roi(frame):
    # x, y, w, h = roi
    # roi_frame = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Apply thresholding
    # _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # # Additional preprocessing
    # kernel = np.ones((3, 3), np.uint8)
    # roi_processed = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)

    cv2.imshow("ROI Processed", roi_gray)
    cv2.waitKey(0)

    # Perform OCR using Tesseract OCR
    extracted_text = pytesseract.image_to_string(roi_gray, config='--psm 6')

    # Postprocess the extracted text to obtain the speed
    speed = process_extracted_text(extracted_text)

    return speed


def process_extracted_text(text):
    # Process the extracted text to obtain the speed
    speed = ''.join(filter(str.isdigit, text))  # Extract only digits from the text
    return speed

image = cv2.imread('../data/audi_speed.png')
h, w = image.shape[:2]
speed = extract_speed_from_roi(image)
print("Extracted Speed:", speed)
