from tracks import *

def create_rotated_templates(template_path, rotations):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    height, width = template.shape
    
    # Initialize list to store rotated templates
    rotated_templates = {}
    
    # Generate rotated templates
    for angle in rotations:
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_template = cv2.warpAffine(template, rotation_matrix, (width, height))
        rotated_templates[angle] = rotated_template
    
    return rotated_templates


def find_steering_angle(frame, templates):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_match_score = -1
    best_angle = 0
    
    for angle, template in templates.items():
        frame_height, frame_width = gray.shape
        template = cv2.resize(template, (frame_width, frame_height))
        # Perform template matching within speedometer area
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, match_score, _, _ = cv2.minMaxLoc(result)
        
        # Update best match score and angle
        if match_score > best_match_score:
            best_match_score = match_score
            best_angle = angle
            
    return best_angle


if __name__ == "__main__":
    # Initialize feature detector
    orb, bf = init_feature_tracking()

    # Initialize template for speedometer
    speedometer_templates = load_template_pyramid("../data/audi_speedometer.png", 2, 1)

    # Define rotations (in degrees) for creating rotated templates
    rotations = list(range(-90, 91, 2))

    # Load template image of steering wheel logo
    steering_template_path = "../data/audi_logo.png"

    # Create rotated templates
    rotated_steering_templates = create_rotated_templates(steering_template_path, rotations)

    # Open video file
    cap = cv2.VideoCapture("../data/audi_gravel_road_footage.mp4")

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
        steering_angle = find_steering_angle(steering_subimage, rotated_steering_templates)
        
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
