import math

import cv2
import numpy as np

def find_speedometer(frame, template, threshold=0.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get the width and height of the template
    h, w = template.shape[:2]
    # Perform template matching
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw rectangle around the best match
    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = (top_left[0] + w // 2, top_left[1] + h // 2)
        return center, top_left, bottom_right

    return None, None, None

# Initialize feature detector
orb = cv2.ORB.create()

# Initialize a matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Open video file
cap = cv2.VideoCapture("../data/audi_road_cropped.mp4")

# Read the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Find keypoints and descriptors in the first frame
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

# Initialize variables to store matched keypoints and their vectors
matched_keypoints = []

# Initialize template for speedometer
template = cv2.imread("../data/speed_template.png", cv2.IMREAD_GRAYSCALE)

orginal_center = None
offset = None
predicted_center = None

cap.set(cv2.CAP_PROP_POS_FRAMES, 1350)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Find the speedometer
    template_center, top_left, bottom_right = find_speedometer(frame, template, threshold=0.5)


    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
    cv2.circle(frame, template_center, 5, (255, 0, 0), -1)
    cv2.circle(frame, top_left, 5, (255, 0, 255), -1)
    cv2.circle(frame, bottom_right, 5, (255, 0, 255), -1)


    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in the current frame
    kp, des = orb.detectAndCompute(gray, None)

    if des is not None:

        # Match descriptors between the current and previous frames
        matches = bf.match(prev_des, des)

        # Initialize variables to store matched keypoints and their vectors for this frame
        vectors_frame = []

        # Process each match
        for match in matches:
            # Get the keypoints for this match
            prev_kp_match = prev_kp[match.queryIdx]
            cur_kp_match = kp[match.trainIdx]
            prev_pt = prev_kp_match.pt
            cur_pt = cur_kp_match.pt

            # Check if the matched keypoint is within 100 pixels of the speedometer center
            if template_center is not None and math.dist(cur_pt, template_center) > 200:
                continue

            cv2.circle(frame, (int(cur_pt[0]), int(cur_pt[1])), 5, (0, 255, 0), -1)

            # Calculate vector from previous point to current point
            vector = np.array([cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1]])

            # Append matched keypoint and vector to lists
            vectors_frame.append(vector)

        # Compute the mean vector for this frame
        if vectors_frame:
            mean_vector = np.mean(vectors_frame, axis=0)

        # Draw vector
        if vectors_frame and mean_vector is not None and prev_pt is not None and template_center is not None:
            if orginal_center is None or offset is None:
                orginal_center = template_center
                offset = template_center

            # Modify the original center by the mean vector
            offset = (int(offset[0] + mean_vector[0]), int(offset[1] + mean_vector[1]))
            predicted_center = offset
            cv2.arrowedLine(frame, orginal_center, offset, (0, 255, 0), 10)


        # Display the matches
        cv2.imshow("Matches", frame)


    # Update previous frame and keypoints
    prev_gray = gray.copy()
    prev_frame = frame.copy()
    prev_kp = kp
    prev_des = des

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
