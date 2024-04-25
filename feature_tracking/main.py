import cv2
import numpy as np
from matplotlib import pyplot as plt

from feature_tracking.alpha_beta_filter import AlphaBeta
from feature_tracking.orientation_2 import find_angle_of_least_inertia, get_steering_angle
from feature_tracking.tracks import init_feature_tracking, load_template_pyramid, find_speedometer, get_road_angle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


def run(start_frame=3000, frames_to_process=500):
    # Initialize feature detector
    orb, bf = init_feature_tracking()

    # Initialize template for speedometer
    speedometer_templates = load_template_pyramid("../data/audi_speedometer.png", 2, 1)

    # Open video file
    cap = cv2.VideoCapture("../data/audi_road_cropped.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # set fps to 30
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Skip to frame x
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize variables
    prev_kp, prev_des = None, None
    speedometer_center = None

    # Init alpha beta filter
    alpha_beta_filter_road_angle = None
    alpha_beta_filter_steering_angle = None

    dt_road = 0
    dt_steering = 0

    wheel_angle_history = []
    road_angle_history = []

    original_wheel_angle = []
    original_road_angle = []

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        dt_steering += 1
        dt_road += 1

        # Find the speedometer
        speedometer_center, kp, des, t_l, b_r = find_speedometer(frame.copy(), orb, bf, prev_kp, prev_des,
                                                                 speedometer_templates,
                                                                 speedometer_center,
                                                                 threshold=0.2, apply_filtering=True, debug=True)

        # Update previous frame and keypoints
        prev_kp = kp
        prev_des = des

        if True:
            cv2.circle(frame, speedometer_center, 5, (0, 255, 0), -1)

        road_angle = get_road_angle(speedometer_center, frame.copy(), debug=True)
        if road_angle is not None:
            original_road_angle.append(road_angle)
        elif len(original_road_angle) > 0:
            original_road_angle.append(original_road_angle[-1])

        if alpha_beta_filter_road_angle is None and road_angle is not None:
            alpha_beta_filter_road_angle = AlphaBeta(alpha=0.04, beta=0.003, initial_position=road_angle)
            dt_road = 0
        elif alpha_beta_filter_road_angle is not None and road_angle is not None:
            road_angle = alpha_beta_filter_road_angle.update(road_angle, dt_road)
            dt_road = 0
        elif alpha_beta_filter_road_angle is not None and road_angle is None:
            road_angle = alpha_beta_filter_road_angle.predict(dt_road)
            dt_road = 0

        if b_r is not None and t_l is not None:
            steering_angle = get_steering_angle(b_r, t_l, frame.copy(), debug=True)
        else:
            steering_angle = wheel_angle_history[-1] if len(wheel_angle_history) > 0 else None

        if steering_angle is not None:
            original_wheel_angle.append(steering_angle)
        elif len(original_wheel_angle) > 0:
            original_wheel_angle.append(original_wheel_angle[-1])

        if alpha_beta_filter_steering_angle is None and steering_angle is not None:
            alpha_beta_filter_steering_angle = AlphaBeta(alpha=0.04, beta=0.001, initial_position=steering_angle)
            dt_steering = 0
        elif alpha_beta_filter_steering_angle is not None and steering_angle is not None:
            steering_angle = alpha_beta_filter_steering_angle.update(steering_angle, dt_steering)
            dt_steering = 0
        elif alpha_beta_filter_steering_angle is not None and steering_angle is None:
            steering_angle = alpha_beta_filter_steering_angle.predict(dt_steering)
            dt_steering = 0

        # Display the angle on the image
        angle_text = f"Steering angle: {steering_angle}"
        cv2.putText(frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Road angle: {road_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Frame", frame)

        if steering_angle is not None and road_angle is not None:
            wheel_angle_history.append(steering_angle)
            road_angle_history.append(road_angle)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(road_angle_history) > frames_to_process:
            break

        print(len(road_angle_history))

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

    return wheel_angle_history, road_angle_history, original_wheel_angle, original_road_angle

if __name__ == "__main__":
    wheel_angle_history, road_angle_history, original_wheel_angle, original_road_angle = run(start_frame=3000, frames_to_process=500)

    wheel_angle_history = np.array(wheel_angle_history)
    road_angle_history = np.array(road_angle_history)

    print(original_wheel_angle)
    print(original_road_angle)

    plt.title("Steering angle vs Road angle (Original)")
    plt.plot(original_wheel_angle, label="Original wheel angle")
    plt.plot(original_road_angle, label="Original road angle")
    plt.legend()
    plt.show()

    plt.title("Steering angle vs Road angle (Filtered)")
    plt.plot(wheel_angle_history, label="Wheel angle")
    plt.plot(road_angle_history, label="Road angle")
    plt.legend()
    plt.show()

    # The phase shift needed to align the road angle with the wheel angle
    # this is because the curve you need to turn is AHEAD of you
    # so you must use past information to predict the steering angle
    # and NOT current information
    HIST_COEF = 100

    road_angle_ts = []
    wheel_angle_ts = []

    for i in range(len(road_angle_history)):
        if i < HIST_COEF:
            continue
        road_angle_ts.append(road_angle_history[i-HIST_COEF:i])
        wheel_angle_ts.append(wheel_angle_history[i])

    road_angle_ts_arr = np.array(road_angle_ts)
    wheel_angle_ts_arr = np.array(wheel_angle_ts)

    print(road_angle_ts_arr.shape, wheel_angle_ts_arr.shape)

    model = SVR()
    model.fit(road_angle_ts_arr, wheel_angle_ts_arr)

    yp = model.predict(road_angle_ts_arr)

    print(yp.shape)

    print("R2", r2_score(wheel_angle_ts_arr, yp))
    print("RMSE", mean_squared_error(wheel_angle_ts_arr, yp, squared=False))


    # evaluate the model again this
    wheel_angle_history, road_angle_history, original_wheel_angle, original_road_angle = run(start_frame=2500, frames_to_process=500)
    road_angle_ts = []
    wheel_angle_ts = []

    for i in range(len(road_angle_history)):
        if i < HIST_COEF:
            continue
        road_angle_ts.append(road_angle_history[i - HIST_COEF:i])
        wheel_angle_ts.append(wheel_angle_history[i])

    road_angle_ts_arr = np.array(road_angle_ts)
    wheel_angle_ts_arr = np.array(wheel_angle_ts)

    yp = model.predict(road_angle_ts_arr)

    print("R2", r2_score(wheel_angle_ts_arr, yp))
    print("RMSE", mean_squared_error(wheel_angle_ts_arr, yp, squared=False))