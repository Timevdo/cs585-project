import cv2
import numpy as np

from feature_tracking.tracks import init_feature_tracking, load_template_pyramid, find_speedometer

if __name__ == "__main__":

    # Initialize feature detector
    orb, bf = init_feature_tracking()

    # Load template image of steering wheel logo
    speedometer_template_path = "../data/honda_speedometer.png"

    # Initialize template for speedometer
    speedometer_templates = load_template_pyramid(speedometer_template_path, 2, 1)
    assert len(speedometer_templates) > 0

    # Open video file
    cap = cv2.VideoCapture("../data/honda_raw_data.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize variables
    prev_kp, prev_des = None, None
    speedometer_center = None

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Find the speedometer
        speedometer_center, kp, des, t_l, b_r = find_speedometer(frame, orb, bf, prev_kp, prev_des,
                                                                 speedometer_templates,
                                                                 speedometer_center,
                                                                 threshold=0.2, debug=False)

        cv2.circle(frame, speedometer_center, 5, (0, 0, 255), -1)
        cv2.imshow("Frame", frame)

        # Number of pixels below speedometer center
        pixels_below_center = 300
        # Width and height of the ROI
        w, h = 300, 300

        # Update previous frame and keypoints
        prev_kp = kp
        prev_des = des

        if speedometer_center is not None:

            # Extract the region of interest (ROI) from the frame
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = frame[speedometer_center[1] + pixels_below_center:speedometer_center[1] + pixels_below_center + h,
                            speedometer_center[0] - w // 2:speedometer_center[0] + w // 2]
            #cv2.imshow("cropped steering", steering_subimage)
            canny_frame = cv2.Canny(frame, 150, 100, apertureSize=3)

            # Detect lines using HoughLinesP
            linesP = cv2.HoughLinesP(canny_frame, 1, np.pi / 180, 50, None, 50, 10)

            # Draw the lines
            if linesP is not None:
                for line in linesP:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(canny_frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

            # Display the frame
            cv2.imshow("Canny", canny_frame)
            cv2.waitKey(1)