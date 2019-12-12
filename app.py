import logging
import time

import edgeiq
import cv2


class YMCAPose(object):
    def __init__(self, pose):
        self.r_wrist_y = pose.key_points[4][1]
        self.r_wrist_x = pose.key_points[4][0]
        self.r_elbow_y = pose.key_points[3][1]
        self.r_elbow_x = pose.key_points[3][0]
        self.l_wrist_y = pose.key_points[7][1]
        self.l_wrist_x = pose.key_points[7][0]
        self.l_elbow_y = pose.key_points[6][1]
        self.l_elbow_x = pose.key_points[6][0]
        self.nose_y = pose.key_points[0][1]
        self.nose_x = pose.key_points[0][0]
        self.neck_y = pose.key_points[1][0]
        self.neck_distance = abs(self.neck_y - self.nose_y)


def arms_overhead(pose):
    if not all([pose.nose_y, pose.r_elbow_y, pose.r_wrist_y, pose.l_elbow_y, pose.l_wrist_y]):
        return False
    return all(y < pose.nose_y for y in (pose.r_elbow_y, pose.r_wrist_y, pose.l_elbow_y, pose.l_wrist_y))


def arms_outward(pose):
    if not all([pose.nose_x, pose.r_elbow_x, pose.r_wrist_x, pose.l_elbow_x, pose.l_wrist_x]):
        return False
    return all([x < pose.nose_x for x in (pose.r_elbow_x, pose.r_wrist_x)] +\
               [x > pose.nose_x for x in (pose.l_elbow_x, pose.l_wrist_x)])


def arms_straight(pose):
    if not all([pose.r_wrist_y, pose.l_wrist_y, pose.r_wrist_x, pose.l_wrist_x, pose.r_elbow_y, pose.l_elbow_y, pose.r_elbow_x, pose.l_elbow_x]):
        return False
    return all([pose.r_wrist_y < pose.r_elbow_y,
                pose.l_wrist_y < pose.l_elbow_y,
                pose.r_wrist_x < pose.r_elbow_x,
                pose.l_wrist_x > pose.l_elbow_x])


def wrists_overhead(pose):
    if not all([pose.nose_y, pose.r_wrist_y, pose.l_wrist_y]):
        return False
    return all(y < pose.nose_y for y in (pose.r_wrist_y, pose.l_wrist_y))


def arms_bent_in(pose):
    if not all([pose.r_wrist_y, pose.l_wrist_y, pose.r_wrist_x, pose.l_wrist_x, pose.r_elbow_y, pose.l_elbow_y, pose.r_elbow_x, pose.l_elbow_x]):
        return False
    return all([pose.r_wrist_y < pose.r_elbow_y,
                pose.l_wrist_y < pose.l_elbow_y,
                pose.r_wrist_x > pose.r_elbow_x,
                pose.l_wrist_x < pose.l_elbow_x])


def height_threshold(pose):
    if not all([pose.nose_y, pose.neck_y]):
        return False
    return pose.neck_distance / 2.0


def wrists_high(pose):
    if not all([pose.r_wrist_y, pose.l_wrist_y, pose.nose_y]):
        return False
    return all([pose.r_wrist_y < (pose.nose_y - height_threshold(pose)),
                pose.l_wrist_y < (pose.nose_y - height_threshold(pose))])


def wrists_low(pose):
    if not all([pose.r_wrist_y, pose.l_wrist_y, pose.nose_y]):
        return False
    return all([pose.r_wrist_y > (pose.nose_y - height_threshold(pose)), pose.l_wrist_y > (pose.nose_y - height_threshold(pose))])


def wrists_left(pose):
    if not all([pose.r_wrist_x, pose.nose_x, pose.l_wrist_x]):
        return False
    return all([pose.r_wrist_x > pose.nose_x, pose.l_wrist_x > pose.nose_x])

def right_wrist_overhead(pose):
    if not all([pose.r_wrist_y, pose.nose_y]):
        return False
    return pose.r_wrist_y < pose.nose_y

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
    pose_estimator.load(engine=edgeiq.Engine.DNN_OPENVINO, accelerator=edgeiq.Accelerator.MYRIAD)

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
                for ind, pose in enumerate(results.poses):
                    app_pose = YMCAPose(pose)

                    if is_a(app_pose):
                        overlay = edgeiq.resize(a_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue
                    if is_m(app_pose):
                        overlay = edgeiq.resize(m_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue
                    if is_y(app_pose):
                        overlay = edgeiq.resize(y_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue
                    if is_c(app_pose):
                        overlay = edgeiq.resize(c_letter, frame.shape[1], frame.shape[0], False)
                        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                        continue

                streamer.send_data(results.draw_poses(frame), text)


                if streamer.check_exit():
                    break
    finally:
        print("Program Ending")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
