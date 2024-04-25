import math

import cv2

from feature_tracking.speedometer_tracker import *

def create_rotated_templates(template_path, rot_delta):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    height, width = template.shape
    
    # Initialize list to store rotated templates
    rotated_templates = {}

    # Because rotation will cause the image to be cropped, we need to scale the image to ensure that the
    # entire image is visible after rotation and there are no black borders.
    c = math.sqrt(math.pow(template.shape[0], 2) + math.pow(template.shape[1], 2))
    scale_factor = min(template.shape[0], template.shape[1]) / c
    new_width = int(scale_factor * template.shape[1])
    new_height = int(scale_factor * template.shape[0])

    center_y = template.shape[0] // 2
    center_x = template.shape[1] // 2

    y_min_crop = center_y - new_height // 2
    y_max_crop = center_y + new_height // 2
    x_min_crop = center_x - new_width // 2
    x_max_crop = center_x + new_width // 2


    # Generate rotated templates
    for angle in range(-90, 91, rot_delta):
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_template = cv2.warpAffine(template, rotation_matrix, (width, height))
        rotated_template = rotated_template[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
        rotated_templates[angle] = rotated_template
    
    return rotated_templates


def find_steering_angle(frame, templates):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_match_score = -1
    best_angle = 0
    match_min_loc = None
    match_max_loc = None
    
    for angle, template in templates.items():
        frame_height, frame_width = gray.shape
        template = cv2.resize(template, (frame_width, frame_height))
        # Perform template matching within speedometer area
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, match_score, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Update best match score and angle
        if match_score > best_match_score:
            best_match_score = match_score
            best_angle = angle
            match_min_loc = min_loc
            match_max_loc = max_loc

    top_left = match_max_loc
    bottom_right = (top_left[0] + templates[best_angle].shape[1], top_left[1] + templates[best_angle].shape[0])
            
    return best_angle, top_left, bottom_right


if __name__ == "__main__":
    # Initialize feature detector
    orb, bf = init_feature_tracking()

    # Load template image of steering wheel logo
    speedometer_template_path = "../data/speed_template.png"

    # Initialize template for speedometer
    steering_template_path = "../data/steering.png"

    # Initialize template for speedometer
    speedometer_templates = load_template_pyramid(speedometer_template_path, 2, 1)
    assert len(speedometer_templates) > 0

    # Create rotated templates
    rotated_steering_templates = create_rotated_templates(steering_template_path, 2)

    # Open video file
    cap = cv2.VideoCapture("../data/audi_road_cropped.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize variables
    prev_kp, prev_des = None, None
    speedometer_center = None

    # Skip to frame 3050
    cap.set(cv2.CAP_PROP_POS_FRAMES, 3050)

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
        steering_subimage = frame[t_l[1]:, t_l[0]:b_r[0]]
        cv2.imshow("cropped steering", steering_subimage)

        # Find steering angle using rotated templates
        steering_angle, top_left, bottom_right = find_steering_angle(steering_subimage, rotated_steering_templates)

        # Draw template
        cv2.rectangle(steering_subimage, top_left, bottom_right, (0, 255, 0), 2)
        
        # Display the angle on the image
        angle_text = f"Estimated steering angle: {steering_angle}"
        cv2.putText(steering_subimage, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # show
        cv2.imshow("cropped steering", steering_subimage)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
