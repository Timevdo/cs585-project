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


def init_feature_tracking():
    # Initialize feature detector
    orb = cv2.ORB.create()
    # Initialize a matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return orb, bf


def find_speedometer(frame, orb, bf, prev_kp, prev_des, speedometer_templates,
                     template_match_history, pt_color_dict, predicted_center, threshold=0.2, debug=False):
    # Find the speedometer
    template_center, top_left, bottom_right = find_speedometer_template_matching(frame, speedometer_templates,
                                                                                 threshold=threshold)
    template_match_history += [template_center]

    # If a match is found, draw it
    if template_center is not None and debug:
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
        cv2.circle(frame, template_center, 5, (255, 0, 0), -1)
        cv2.circle(frame, top_left, 5, (255, 0, 255), -1)
        cv2.circle(frame, bottom_right, 5, (255, 0, 255), -1)

    # Decide whether to set to predicted center to the template center if the template center is found and
    # we are confident in the match. If the template center has stayed the same for the last ~10 frames
    # TODO: Make hyper parameters adjustable
    # TODO: make the max jump distance allow depend on how long its been since the last match
    if template_center is not None and len(template_match_history) > 10:
        if all([x is not None and math.dist(template_center, x) < 25 for x in template_match_history[-10:]]):
            predicted_center = template_center

    temp_history = template_match_history[-10:]
    template_match_history.clear()
    template_match_history.extend(temp_history)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in the current frame
    kp, des = orb.detectAndCompute(gray_frame, None)

    if des is not None and prev_des is not None and prev_kp is not None and kp is not None:

        # Match descriptors between the current and previous frames
        matches = bf.match(prev_des, des)

        # Initialize variables to store matched keypoints and their vectors for this frame
        vectors_frame = []
        matched_kps = []

        # Process each match
        for match in matches:
            # Get the keypoints for this match
            prev_kp_match = prev_kp[match.queryIdx]
            cur_kp_match = kp[match.trainIdx]
            prev_pt = prev_kp_match.pt
            cur_pt = cur_kp_match.pt
            # If we don't have a predicted center, skip
            if predicted_center is None:
                continue
            # Check if the matched keypoint is outside the template bounding box
            # skip if it is
            if cur_pt[1] < predicted_center[1]:
                continue
            matched_kps.append((prev_kp_match, cur_kp_match))

        # Filter out weak matches by removing matches with a response less than the 0.5 std below the mean response
        # (responses are an indicator of the quality of the match)
        responses = [kp.response for kp in kp]
        mean_response = np.mean(responses)
        std_response = np.std(responses)
        matched_kps = [(prev_kp, cur_kp) for prev_kp, cur_kp in matched_kps
                       if cur_kp.response > mean_response - (std_response * 0.5)]

        for matched_kp in matched_kps:
            prev_kp_match, cur_kp_match = matched_kp
            prev_pt = prev_kp_match.pt
            cur_pt = cur_kp_match.pt

            # Match keypoints with colors such that the same keypoint will have the same color
            if prev_pt in pt_color_dict:
                color = pt_color_dict[prev_pt]
                del pt_color_dict[prev_pt]
                pt_color_dict[cur_pt] = color
            else:
                pt_color_dict[cur_pt] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Visualize the matched keypoints
            if debug:
                cv2.circle(frame, (int(cur_pt[0]), int(cur_pt[1])), 5, pt_color_dict[cur_pt], -1)

            # Calculate vector from previous point to current point
            vector = np.array([cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1]])

            # Append matched keypoint and vector to lists
            vectors_frame.append(vector)

        mean_vector = None

        # Compute the mean vector from all points that are not outliers (becaused on their norm)
        if vectors_frame:
            # Remove outliers (based on std)
            norms = np.linalg.norm(vectors_frame, axis=1)
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            vectors_frame = [v for i, v in enumerate(vectors_frame) if norms[i] < mean_norm + (1.5 * std_norm)]
            # Compute the mean vector
            mean_vector = np.mean(vectors_frame, axis=0)

        # Draw vector
        if vectors_frame and mean_vector is not None and predicted_center is not None:
            # Modify the original center by the mean vector
            offset = (int(predicted_center[0] + mean_vector[0]), int(predicted_center[1] + mean_vector[1]))
            if debug:
                # The gray is the old center, the white is the offset based on the optical flow
                cv2.circle(frame, predicted_center, 5, (100, 100, 100), -1)
                cv2.circle(frame, offset, 5, (255, 255, 255), -1)
            predicted_center = offset

        # Display the matches
        if debug:
            cv2.imshow("Matches", frame)

    return predicted_center, kp, des


if __name__ == "__main__":
    # Initialize feature detector
    orb, bf = init_feature_tracking()

    # Initialize template for speedometer
    speedometer_templates = load_template_pyramid("../data/audi_speedometer.png", 2, 1)

    # Open video file
    cap = cv2.VideoCapture("../data/audi_gravel_road_footage.mp4")

    # Initialize variables
    template_match_history = []
    pt_color_dict = {}
    prev_kp, prev_des = None, None
    predicted_center = None

    # Skip to frame 3050
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 3050)

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Find the speedometer
        predicted_center, kp, des = find_speedometer(frame, orb, bf, prev_kp, prev_des, speedometer_templates,
                                                     template_match_history, pt_color_dict, predicted_center,
                                                     threshold=0.2, debug=True)

        # Update previous frame and keypoints
        prev_kp = kp
        prev_des = des

        # Flood fill 50 pixels above the predicted center
        # to find the road
        if predicted_center is not None and predicted_center[1] - 150 > 0:
            frame = cv2.medianBlur(frame, 5)
            cv2.floodFill(frame, None, (predicted_center[0], predicted_center[1] - 150), (255, 0, 0), loDiff=(2, 2, 2),
                          upDiff=(2, 2, 2))
            # mark the flood fill center
            cv2.circle(frame, (predicted_center[0], predicted_center[1] - 150), 5, (0, 255, 0), -1)

        # show
        cv2.imshow("Road", frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
