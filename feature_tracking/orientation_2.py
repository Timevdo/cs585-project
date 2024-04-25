from tracks import *
import os


def calc_area(img):
    return np.count_nonzero(img)


def find_centroid(img):
    area = calc_area(img)
    m10 = np.sum(np.where(img > 0)[0])
    m01 = np.sum(np.where(img > 0)[1])

    # calculate the centroid
    x = m10 / area
    y = m01 / area

    return x, y


def find_angle_of_least_inertia(img):
    area = calc_area(img)
    x, y = find_centroid(img)

    # calculate the centroid
    a = np.sum((np.where(img > 0)[0] - x) ** 2)
    b = 2 * np.sum((np.where(img > 0)[0] - x) * (np.where(img > 0)[1] - y))
    c = np.sum((np.where(img > 0)[1] - y) ** 2)

    # calculate the angle of least inertia
    theta = 0.5 * np.arctan2(b, a - c)

    # draw the angle line with midpoint at centroid (using polar to cartesian conversion_
    axis_len = 100
    x1 = int(x + (axis_len * np.cos(theta)))
    y1 = int(y + (axis_len * np.sin(theta)))
    x2 = int(x - (axis_len * np.cos(theta)))
    y2 = int(y - (axis_len * np.sin(theta)))

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # draw line of least inertia
    img = cv2.line(img, (y1, x1), (y2, x2), (0, 255, 0), 2)
    # draw marker at the centroid
    img = cv2.drawMarker(img, (int(y), int(x)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10,
                        thickness=2)
    cv2.putText(img, f'{theta}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Angle of Least Inertia Image", img)

    # steering_angle = (np.pi / 2 - theta) if (theta > 0) else (-1 * (np.pi / 2 - abs(theta)))
    steering_angle = np.sign(theta) * (np.pi / 2 - abs(theta))
    
    return np.rad2deg(steering_angle)

def get_steering_angle(speed_bottom_right, speed_top_left, frame, debug=False):
    # Extract the region of interest (ROI) from the frame
    steering_subimage = frame[speed_bottom_right[1]:, speed_top_left[0]:speed_bottom_right[0]]

    if 0 in steering_subimage.shape:
        return None

    # Convert image to grayscale
    gray = cv2.cvtColor(steering_subimage, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(gray, 50, 150)

    # Apply dilation to enhance the edges
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        cv2.imshow("Dilated Edges", dilated_edges)

    # If no contours, continue
    if len(contours) == 0:
        return None

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask image for the largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the edge-detected image
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Calculate angle of least inertia for the masked edges
    steering_angle = find_angle_of_least_inertia(masked_edges)

    return steering_angle


if __name__ == "__main__":
    # Initialize feature detector
    orb, bf = init_feature_tracking()

    # Load template image of steering wheel logo
    speedometer_template_path = "../data/audi_speedometer.png"

    if not os.path.exists(speedometer_template_path):
        print("Error: Could not find speedometer template.")
        exit()

    # Initialize template for speedometer
    speedometer_templates = load_template_pyramid(speedometer_template_path, 2, 1)
    assert len(speedometer_templates) > 0

    # Open video file
    cap = cv2.VideoCapture("../data/audi_gravel_road_footage.mp4")

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
        speedometer_center, kp, des, t_l, b_r = find_speedometer(frame, orb, bf, prev_kp, prev_des, speedometer_templates,
                                                    speedometer_center,
                                                    threshold=0.2, debug=False)

        # Update previous frame and keypoints
        prev_kp = kp
        prev_des = des

        # Extract the region of interest (ROI) from the frame
        steering_subimage = frame[b_r[1]:, t_l[0]:b_r[0]]

        if 0 in steering_subimage.shape:
            continue

        # Convert image to grayscale
        gray = cv2.cvtColor(steering_subimage, cv2.COLOR_BGR2GRAY)

        # Use Canny edge detection to find edges
        edges = cv2.Canny(gray, 50, 150)

        # Apply dilation to enhance the edges
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours, continue
        if len(contours) == 0:
            continue

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask image for the largest contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the edge-detected image
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

        # Calculate angle of least inertia for the masked edges
        steering_angle = find_angle_of_least_inertia(masked_edges)

        # Display the angle on the image
        angle_text = f"Steering angle: {steering_angle}"
        cv2.putText(steering_subimage, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # show
        cv2.imshow("steering angle", steering_subimage)
    
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()