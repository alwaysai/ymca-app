import logging
import time

import edgeiq
import cv2


def arms_overhead(pose):
    return all(y < pose.key_points['Nose'].y for y in (pose.key_points['Right Elbow'].y, pose.key_points['Right Wrist'].y, pose.key_points['Left Elbow'].y, pose.key_points['Left Wrist'].y))


def arms_outward(pose):
    return all([x < pose.key_points['Nose'].x for x in (pose.key_points['Right Elbow'].x, pose.key_points['Right Wrist'].x)] +
               [x > pose.key_points['Nose'].x for x in (pose.key_points['Left Elbow'].x, pose.key_points['Left Wrist'].x)])


def arms_straight(pose):
    return all([pose.key_points['Right Wrist'].y < pose.key_points['Right Elbow'].y,
                pose.key_points['Left Wrist'].y < pose.key_points['Left Elbow'].y,
                pose.key_points['Right Wrist'].x < pose.key_points['Right Elbow'].x,
                pose.key_points['Left Wrist'].x > pose.key_points['Left Elbow'].x])


def wrists_overhead(pose):
    return all(y < pose.key_points['Nose'].y for y in (pose.key_points['Right Wrist'].y, pose.key_points['Left Wrist'].y))


def arms_bent_in(pose):
    return all([pose.key_points['Right Wrist'].y < pose.key_points['Right Elbow'].y,
                pose.key_points['Left Wrist'].y < pose.key_points['Left Elbow'].y,
                pose.key_points['Right Wrist'].x > pose.key_points['Right Elbow'].x,
                pose.key_points['Left Wrist'].x < pose.key_points['Left Elbow'].x])


def height_threshold(pose):
    return pose.key_points['Neck'].y - pose.key_points['Nose'].y


def wrists_high(pose):
    return all([pose.key_points['Right Wrist'].y < (pose.key_points['Nose'].y - height_threshold(pose)),
                pose.key_points['Left Wrist'].y < (pose.key_points['Nose'].y - height_threshold(pose))])


def wrists_low(pose):
    return all([pose.key_points['Right Wrist'].y > (pose.key_points['Nose'].y - height_threshold(pose)), pose.key_points['Left Wrist'].y > (pose.key_points['Nose'].y - height_threshold(pose))])


def wrists_left(pose):
    return all([pose.key_points['Right Wrist'].x > pose.key_points['Nose'].x, pose.key_points['Left Wrist'].x > pose.key_points['Nose'].x])


def right_wrist_overhead(pose):
    return pose.key_points['Right Wrist'].y < pose.key_points['Nose'].y


def is_y(pose):
    """Determines if the pose is a Y pose"""
    return all([arms_overhead(pose), arms_outward(pose), arms_straight(pose)])


def is_a(pose):
    """Determines if the pose is an A."""
    return all([arms_overhead(pose), arms_outward(pose), arms_bent_in(pose), wrists_high(pose)])


def is_m(pose):
    """Determines if the pose is an A."""
    return all([wrists_overhead(pose), arms_outward(pose), arms_bent_in(pose), wrists_low(pose)])


def is_c(pose):
    """Determines if the pose is a C"""
    return all([wrists_left(pose), right_wrist_overhead(pose)])


def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(engine=edgeiq.Engine.DNN)

    y_letter = cv2.imread('y_letter.png')
    m_letter = cv2.imread('m_letter.jpg')
    c_letter = cv2.imread('c_letter.jpeg')
    a_letter = cv2.imread('a_letter.jpg')

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)

            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                # Generate text to display on streamer
                text = [""]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                for ind, pose in enumerate(results.poses):
                    if not all([pose.key_points['Right Wrist'].y, pose.key_points['Left Wrist'].y,
                                pose.key_points['Right Elbow'].y, pose.key_points['Left Elbow'].y,
                                pose.key_points['Nose'].x, pose.key_points['Neck'].y]):
                        continue
                    if is_a(pose):
                        overlay = edgeiq.resize(a_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue
                    if is_m(pose):
                        overlay = edgeiq.resize(m_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue
                    if is_y(pose):
                        overlay = edgeiq.resize(y_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue
                    if is_c(pose):
                        overlay = edgeiq.resize(c_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue

                streamer.send_data(results.draw_poses(frame), text)

                if streamer.check_exit():
                    break
    finally:
        print("Program Ending")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
