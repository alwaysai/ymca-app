import logging
import time

import edgeiq
import cv2


def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(engine=edgeiq.Engine.DNN_OPENVINO, accelerator=edgeiq.Accelerator.MYRIAD)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    y_letter = cv2.imread('letter_y.png')
    m_letter = cv2.imread('m_letter.jpg')
    c_letter = cv2.imread('c_letter.jpeg')
    a_letter = cv2.imread('a_letter.jpg')

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                # Generate text to display on streamer
                text = [""]
                for ind, pose in enumerate(results.poses):
                    right_wrist_y = pose.key_points[4][1]
                    right_wrist_x = pose.key_points[4][0]
                    right_elbow_y = pose.key_points[3][1]
                    right_elbow_x = pose.key_points[3][0]
                    left_wrist_y = pose.key_points[7][1]
                    left_wrist_x = pose.key_points[7][0]
                    left_elbow_y = pose.key_points[6][1]
                    left_elbow_x = pose.key_points[6][0]
                    nose_y = pose.key_points[0][1]
                    nose_x = pose.key_points[0][0]
                    neck_y = pose.key_points[1][0]
                    if nose_y != -1 and neck_y != -1:
                        neck_distance = neck_y - nose_y
                    else:
                        neck_distance = 0
                    if right_wrist_y != -1 and left_wrist_y != -1 and nose_y != -1 and left_elbow_y != -1 and right_elbow_y != -1 and neck_distance > 0:
                        if right_wrist_y < nose_y and left_wrist_y < nose_y and right_wrist_x > right_elbow_x and left_wrist_x < left_elbow_x:
                            if right_wrist_y < (nose_y - neck_distance / 3.0) and left_wrist_y < (nose_y - neck_distance / 3.0):
                                print("----------A!-------------")
                                overlay = edgeiq.resize(a_letter, frame.shape[1], frame.shape[0], False)
                                cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                                continue
                            elif (nose_y - neck_distance) < right_wrist_y and (nose_y - neck_distance) < left_wrist_y:
                                print("----------M!-------------")
                                overlay = edgeiq.resize(m_letter, frame.shape[1], frame.shape[0], False)
                                cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                                continue
                    if right_wrist_y != -1 and left_wrist_y != -1 and nose_y != -1 and right_elbow_x and left_elbow_x and right_wrist_x and left_wrist_x:
                        if right_wrist_y < nose_y and left_wrist_y < nose_y and right_wrist_x < right_elbow_x and left_wrist_x > left_elbow_x:
                            print("----------Y!-------------")
                            overlay = edgeiq.resize(y_letter, frame.shape[1], frame.shape[0], False)
                            cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                            continue
                    if left_wrist_x != -1 and nose_x != -1 and left_wrist_y != -1 and nose_y != -1 and right_wrist_y != -1 and nose_x != -1:
                        if right_wrist_x > nose_x and right_wrist_y < nose_y and left_wrist_x > nose_x:
                            print("----------C!-------------")
                            overlay = edgeiq.resize(c_letter, frame.shape[1], frame.shape[0], False)
                            cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
                            continue

                streamer.send_data(results.draw_poses(frame), text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


    # fps = edgeiq.FPS()

    # with edgeiq.Streamer() as streamer:
        # Allow Webcam to warm up
    # time.sleep(2.0)
    # fps.start()

    # y_letter = cv2.imread('letter_y.png')
    # image = cv2.imread('y_person.jpg')
    # results = pose_estimator.estimate(image)
    # pose = results.poses[0]
    # right_wrist_y = pose.key_points[4][1]
    # left_wrist_y = pose.key_points[7][1]
    # nose_y = pose.key_points[0][1]
    # if right_wrist_y < nose_y and left_wrist_y < nose_y:
    #     print("----------Y!-------------")
    #     overlay = edgeiq.resize(y_letter, image.shape[1], image.shape[0], False)
    #     cv2.addWeighted(image, 0.4, overlay, 0.6, 0, image)
    # cv2.imwrite("woot.png", image)
    # print("{} {} {}".format(left_wrist_y, right_wrist_y, nose_y))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
