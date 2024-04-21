import cv2
import numpy as np

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

orginal_center = None
offset = None

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in the current frame
    kp, des = orb.detectAndCompute(gray, None)

    if des is not None:

        # Match descriptors between the current and previous frames
        matches = bf.match(prev_des, des)

        # Initialize variables to store matched keypoints and their vectors for this frame
        matched_keypoints_frame = []
        vectors_frame = []

        # Process each match
        for match in matches:
            # Get the keypoints for this match
            prev_pt = prev_kp[match.queryIdx].pt
            cur_pt = kp[match.trainIdx].pt

            # Calculate vector from previous point to current point
            vector = np.array([cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1]])

            # Append matched keypoint and vector to lists
            matched_keypoints_frame.append((prev_pt, cur_pt))
            vectors_frame.append(vector)

        # Compute the mean vector for this frame
        if vectors_frame:
            mean_vector = np.mean(vectors_frame, axis=0)

        # Store matched keypoints and vectors for this frame
        matched_keypoints.append(matched_keypoints_frame)

        # Draw vector
        if vectors_frame and mean_vector is not None and prev_pt is not None:
            if orginal_center is None or offset is None:
                orginal_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                offset = (frame.shape[1] // 2, frame.shape[0] // 2)

            # Modify the original center by the mean vector
            offset = (int(offset[0] + mean_vector[0]), int(offset[1] + mean_vector[1]))

            cv2.arrowedLine(frame, orginal_center, offset, (0, 255, 0), 10)


        # Display the matches
        cv2.imshow("Matches", frame)


    # Update previous frame and keypoints
    prev_gray = gray.copy()
    prev_frame = frame.copy()
    prev_kp = kp
    prev_des = des

    # Check for key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
