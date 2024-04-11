import cv2

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

