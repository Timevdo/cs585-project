import cv2
import numpy as np

def crop_video(input_video_path, output_video_path, start_time, end_time):
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Get the frame rate and frame size
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate start and end frame numbers based on time stamps
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Read until video is completed
    while True:
        ret, frame = video_capture.read()

        # Break the loop if video is completed
        if not ret:
            break

        # Write frame to the output video if it falls within the specified time range
        if start_frame <= video_capture.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            output_video.write(frame)

        # Break the loop if end frame is reached
        if video_capture.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
            break

    # Release video capture and writer
    video_capture.release()
    output_video.release()

def segment_hands(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred_frame = cv2.GaussianBlur(hsv_frame, (11, 11), 0)

    lower = np.array([0, 10, 60], dtype = "uint8") 
    upper = np.array([20, 150, 255], dtype = "uint8")

    mask = cv2.inRange(blurred_frame, lower, upper)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    sorted_contours = sorted(contours, key=cv2.contourArea)

    hand1 = sorted_contours[-1]
    hand2 = sorted_contours[-2]

    def center_of_contours(c):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return (cX, cY)
    
    c1 = center_of_contours(hand1)
    c2 = center_of_contours(hand2)

    if (c1[0] > c2[0]):
        return np.array((c2, c1))
    return np.array((c1, c2))

def angle_of_wheel_line(c1, c2):
    x1, y1 = c1
    x2, y2 = c2

    return np.rad2deg(np.arctan2((y2 - y1), (x2 - x1)))