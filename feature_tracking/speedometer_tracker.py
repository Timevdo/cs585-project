import math
import random

import cv2
import numpy as np

from feature_tracking.alpha_beta_filter import AlphaBeta


def find_speedometer_template_matching(frame, templates, predicted_center, threshold=0.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # crop the frame to the most likely area of the speedometer (to save performance)
    # 80% of the time; not 100% to allow for correction in case predicted_center is wrong
    # we want to be able to recover from a bad prediction
    #if predicted_center is not None and random.uniform(0, 1) > 0.8:
    #    gray = gray[
    #               max(predicted_center[1] - 250, 0):min(predicted_center[1] + 250,
    #               frame.shape[0]), max(predicted_center[0] - 250, 0):min(predicted_center[0] + 250,
    #               frame.shape[1])
    #           ]

    best_match = None

    for template in templates:
        # Get the width and height of the template
        h, w = template.shape[:2]
        assert h > 0 and w > 0
        assert h < gray.shape[0] and w < gray.shape[1]
        # Perform template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

        # Find the location of the best match
        _, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = (top_left[0] + w // 2, top_left[1] + h // 2)

        if best_match is None or max_val > best_match[0]:
            best_match = (max_val, center, top_left, bottom_right, template)

    # Draw rectangle around the best match
    if best_match[0] >= threshold:
        return best_match[1], best_match[2], best_match[3], best_match[4]

    return None, None, None, None


def load_template_pyramid(template_path, down_levels, up_levels, scale_factor=0.9):
        assert 0 < scale_factor < 1
        # TODO: likely possible to create higher quality templates using https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html
        #  This using laplacian pyramids to create a more accurate template true to the edges of the original image
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


prev_frame = None
pt_color_dict = {}
template_match_history = []

alpha_beta_filter_speedometer = None
speed_dt = 0

def find_speedometer(frame, orb, bf, prev_kp, prev_des, speedometer_templates,
                     predicted_center, threshold=0.2, apply_filtering=True, debug=False):

    global prev_frame
    global pt_color_dict
    global template_match_history

    # Find the speedometer
    template_center, top_left, bottom_right, matched_template = find_speedometer_template_matching(frame, speedometer_templates, predicted_center,
                                                                                 threshold=threshold)
    template_match_history += [template_center]

    # If a match is found, draw it for debug
    if template_center is not None and debug:
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
        cv2.circle(frame, template_center, 5, (255, 0, 0), -1)
        cv2.circle(frame, top_left, 5, (255, 0, 255), -1)
        cv2.circle(frame, bottom_right, 5, (255, 0, 255), -1)

    # Decide whether to set to predicted center to the template center if the template center is found and
    # we are confident in the match. If the template center has stayed the same for the last ~30 frames
    # TODO: Make hyper parameters adjustable
    # TODO: make the max jump distance allow depend on how long its been since the last match
    trust_template_match = False
    if template_center is not None and len(template_match_history) >= 5:
        if all([x is not None and math.dist(template_center, x) < 10 for x in template_match_history[-5:]]):
            predicted_center = template_center
            trust_template_match = True

    temp_history = template_match_history[-5:]
    template_match_history.clear()
    template_match_history.extend(temp_history)

    # Mask the frame to the region of interest (which is everything below the top left corner of the speedometer)
    if predicted_center is not None and matched_template is not None:
        frame[:predicted_center[1] - matched_template.shape[0] // 2, :] = (0, 0, 0)

    # Find keypoints and descriptors in the current frame
    kp, des = orb.detectAndCompute(frame, None)

    if des is not None and prev_des is not None and prev_kp is not None and kp is not None:

        # Match descriptors between the current and previous frames
        matches = bf.match(prev_des, des)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Initialize variables to store matched keypoints and their vectors for this frame
        vectors_frame = []

        # Process each match
        for match in matches:
            # Get the keypoints for this match
            prev_kp_match = prev_kp[match.queryIdx]
            cur_kp_match = kp[match.trainIdx]
            # Get the points for this match
            prev_pt = prev_kp_match.pt
            cur_pt = cur_kp_match.pt

            # If we don't have a predicted center, skip
            if predicted_center is None:
                continue

            # Find the vector from the old point to the new point
            vector = np.array([cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1]])
            vectors_frame.append(vector)

            # Visualize the matched keypoints
            if debug:
                cv2.circle(frame, (int(cur_pt[0]), int(cur_pt[1])), 3, (0, 255, 0), -1)

        # Compute the mean vector and remove outliers
        mean_vector = None
        if vectors_frame:
            # Remove outliers (based on std)
            norms = np.linalg.norm(vectors_frame, axis=1)
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            vectors_frame = [v for i, v in enumerate(vectors_frame) if norms[i] < mean_norm + (1. * std_norm)]
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
                # Show the overall optical flow vector for the frame
                cv2.arrowedLine(frame, (frame.shape[1] // 2, frame.shape[0] // 2),
                                (int((frame.shape[1] / 2) + mean_vector[0] * 25),
                                 int(frame.shape[0] / 2 + mean_vector[1] * 25)),
                                (0, 255, 0), 2)
            # only updated the predicted center if we don't trust the template match
            if not trust_template_match:
                predicted_center = offset

        # Display the matches
        if debug:
            cv2.imshow("Matches 2", frame)

    prev_frame = frame

    # Apply filtering
    if apply_filtering:
        global speed_dt
        global alpha_beta_filter_speedometer
        speed_dt += 1
        if alpha_beta_filter_speedometer is None and predicted_center is not None:
            alpha_beta_filter_speedometer = AlphaBeta(alpha=0.2, beta=0.005, initial_position=predicted_center)
            speed_dt = 0
        elif alpha_beta_filter_speedometer is not None and predicted_center is not None:
            predicted_center = np.array(predicted_center)
            predicted_center = alpha_beta_filter_speedometer.update(predicted_center, speed_dt)
            predicted_center = predicted_center.astype(np.int32)
            predicted_center = tuple(predicted_center)
            speed_dt = 0
        elif alpha_beta_filter_speedometer is not None and predicted_center is None:
            predicted_center = alpha_beta_filter_speedometer.predict(speed_dt)
            predicted_center = predicted_center.astype(np.int32)
            predicted_center = tuple(predicted_center)
            speed_dt = 0

    return predicted_center, kp, des, top_left, bottom_right

def get_road_angle(speedometer_center, frame, debug):
    road_angle = None
    # Flood fill 150 pixels above the predicted center
    # to find the road
    if speedometer_center is not None and speedometer_center[1] - 150 > 0:
        # road_detect = cv2.medianBlur(frame, 7) # very expensive and doesnt seem to improve results by THAT much
        road_detect = frame.copy()

        masks = []
        for i in range(-120, 121, 10):
            _, _, new_mask, _ = cv2.floodFill(road_detect, None, (speedometer_center[0] + i, speedometer_center[1] - 150),
                                          (255, 0, 0), loDiff=(3, 3, 3), upDiff=(2, 2, 2))
            masks.append(new_mask)

        # Combine masks
        mask = np.zeros_like(masks[0])
        for m in masks:
            mask = cv2.bitwise_or(mask, m)
        # Dilate mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # apply mask on top of the frame
        mask = mask[:frame.shape[0], :frame.shape[1]]
        frame[mask != 0] = (0, 0, 0)

        # show mask as white
        visual_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        visual_mask[mask != 0] = (255, 255, 255)
        if debug:
            cv2.imshow("Mask", visual_mask)

        # Invert the mask for contour detection
        mask[mask == 0] = 255
        mask[mask == 1] = 0

        # Make countours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw the contours with the largest area
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea)
            # the largest contour is the whole image, so we want the second largest
            # which is just the road
            main_contour = sorted_contours[-2] if len(sorted_contours) > 1 else sorted_contours[-1]
            cv2.drawContours(frame, [main_contour], -1, (0, 0, 255), 2)


            # Find the top most point of the contour
            top_most_point = tuple(main_contour[main_contour[:, :, 1].argmin()][0])
            # Find all pixels with in 5 pixels of the top most point
            close_points = [point for point in main_contour if abs(point[0][1] - top_most_point[1]) < 5]
            for cp in close_points:
                cv2.circle(frame, tuple(cp[0]), 5, (0, 255, 0), -1)
            # Find the average x value of the close points
            avg_x = np.mean([cp[0][0] for cp in close_points])
            #cv2.circle(frame, (int(avg_x), top_most_point[1]), 5, (255, 0, 0), -1)
            #cv2.imshow("Road", frame)
            #return
            # mark the flood fill seeds
            for i in range(-120, 121, 10):
                cv2.circle(frame, (speedometer_center[0] + i, speedometer_center[1] - 150), 3, (255, 0, 0), -1)

            # Calculate road angle
            x1, y1 = (speedometer_center[0], speedometer_center[1] - 150)
            x2, y2 = int(avg_x), top_most_point[1]

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            road_angle = np.rad2deg(np.arctan2((y2 - y1), (x2 - x1)))
            cv2.putText(frame, f"Road Angle: {road_angle:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show
    if debug:
        cv2.imshow("Road", frame)

    return road_angle