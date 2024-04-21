import math
import random

import cv2
import numpy as np


def find_speedometer_template_matching(frame, templates, threshold=0.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_match = None

    for template in templates:
        # Get the width and height of the template
        h, w = template.shape[:2]
        # Perform template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

        # Find the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = (top_left[0] + w // 2, top_left[1] + h // 2)

        if best_match is None or max_val > best_match[0]:
            best_match = (max_val, center, top_left, bottom_right)

    # Draw rectangle around the best match
    if best_match[0] >= threshold:
        return best_match[1], best_match[2], best_match[3]

    return None, None, None


def load_template_pyramid(template_path, down_levels, up_levels, scale_factor=0.9):
    assert 0 < scale_factor < 1

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template_pyramid = [template]
    for i in range(1, down_levels + 1):
        template = cv2.resize(template, (
            int(template.shape[1] * scale_factor ** down_levels),
            int(template.shape[0] * scale_factor ** down_levels)
        ))
        template_pyramid.append(template)
    for i in range(1, up_levels + 1):
        template = cv2.resize(template, (
            int(template.shape[1] / (scale_factor ** down_levels)),
            int(template.shape[0] / (scale_factor ** down_levels))
        ))
        template_pyramid.append(template)
    return template_pyramid


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

template_match_history = []

# Initialize template for speedometer
speedometer_templates = load_template_pyramid("../data/speed_template.png", 2, 1)

orginal_center = None
offset = None
predicted_center = None

cap.set(cv2.CAP_PROP_POS_FRAMES, 3050)

pt_color_dict = {}

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Find the speedometer
    template_center, top_left, bottom_right = find_speedometer_template_matching(frame, speedometer_templates, threshold=0.0)
    template_match_history += [template_center]

    # If a match is found, draw it
    if template_center is not None:
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
        cv2.circle(frame, template_center, 5, (255, 0, 0), -1)
        cv2.circle(frame, top_left, 5, (255, 0, 255), -1)
        cv2.circle(frame, bottom_right, 5, (255, 0, 255), -1)

    # Decide whether to set to predicted center to the template center if the template center is found and
    # we are confident in the match. If the template center has stayed the same for the last ~10 frames
    # TODO: Make hyper parameters adjustable
    # TODO: make the max jump distance allow depend on how long its been since the last match
    if template_center is not None and len(template_match_history) > 10:
        if all([x is not None and math.dist(template_center, x) < 15 for x in template_match_history[-5:]]):
            predicted_center = template_center
    template_match_history = template_match_history[-10:]

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

            if prev_pt in pt_color_dict:
                color = pt_color_dict[prev_pt]
                del pt_color_dict[prev_pt]
                pt_color_dict[cur_pt] = color
            else:
                pt_color_dict[cur_pt] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # If we don't have a predicted center, skip
            if predicted_center is None:
                continue

            # Check if the matched keypoint is outside the template bounding box
            # skip if it is
            if cur_pt[1] < predicted_center[1]:
                continue

            cv2.circle(frame, (int(cur_pt[0]), int(cur_pt[1])), 5, pt_color_dict[cur_pt], -1)

            # Calculate vector from previous point to current point
            vector = np.array([cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1]])

            # Append matched keypoint and vector to lists
            vectors_frame.append(vector)

        # Compute the mean vector from the top 25% of the vectors that moved the least
        if vectors_frame:
            # Sort vectors by magnitude
            vectors_frame.sort(key=lambda x: np.linalg.norm(x))
            # Get the top 25% of the vectors that moved the least
            vectors_frame = vectors_frame[:int(len(vectors_frame) * 0.25)]
            # Compute the mean vector
            mean_vector = np.mean(vectors_frame, axis=0)

        # Draw vector
        if vectors_frame and mean_vector is not None and prev_pt is not None and predicted_center is not None:
            # Modify the original center by the mean vector
            offset = (int(predicted_center[0] + mean_vector[0]), int(predicted_center[1] + mean_vector[1]))
            cv2.arrowedLine(frame, predicted_center, offset, (255, 0, 255), 10)
            predicted_center = offset

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
