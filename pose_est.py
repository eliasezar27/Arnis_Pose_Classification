import time

import mediapipe as mp
# from strikes import strike, joint_angles
from pose_grade import strike_grade, joint_angles
from strikes import strike
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import cv2
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

arnis_poses = [" ", "PUGAY", "HANDA",
               "Left Temple Strike", "Right Temple Strike", "Left Shoulder Strike", "Right Shoulder Strike",
               "Stomach Thrust", "Left Chest Thrust", "Right Chest Thrust", "Right Leg Strike",
               "Left Leg Strike", "Left Eye Thrust", "Right Eye Thrust", "Crown Strike",                        # STRIKES
               "Left Temple Block", "Right Temple Block", "Left Shoulder Block", "Right Shoulder Block",
               "Stomach Thrust Block", "Left Chest Block", "Right Chest Block", "Right Leg Block",
               "Left Leg Block", "Left Eye Block", "Right Eye Block", "Rising Block"]


# Detection function for the arnis baston
@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# Method for classifying arnis poses {added: key}
def pose_det(frame, model, shwSkltn, s_angle, key=1, grading=False):
    point_baston = -1
    # Uncomment flip to extract angles, visually
    # frame = cv2.flip(frame, 1)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    grade = 0

    # Detect Baston
    frame, bboxList = det_baston(frame, model)
    h, w, c = frame.shape

    # Detect body keypoints directly from re-RGB frame
    results = pose.process(imgRGB)

    # print(results.pose_landmarks)
    joints = {}
    # Check if there's a detection
    if results.pose_landmarks:

        if shwSkltn:
            # Draw landmark keypoints with edges
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Iterates thru all the landmarks using joints id and its corresponding coordinates
        for jt_id, lm in enumerate(results.pose_landmarks.landmark):
            # print(jt_id, lm)

            # Get the x and y coordinates by multiplying the results to the image's dimension and its lm thresholds
            cx, cy, thr = int(lm.x * w), int(lm.y * h), lm.visibility

            # if the threshold is > 50%
            if thr > 0.5:
                # then the joints and its coordinates is added to the joints dictionary
                joints[jt_id] = (cx, cy)
                # print(jt_id, cx, cy, 'th: ', thr)

    if not grading:
        # Apply pose classification method
        label, point_baston = strike(joints, bboxList)
    else:
        label = arnis_poses[key]

    # Add text colors for each of the strikes and blocks
    color_text = (255, 0, 0) if 'Block' in label else (0, 255, 0)

    if grading:
        tm_now = time.time()
        dat_tm = time.localtime(tm_now)
        print(dat_tm.tm_sec)
        if dat_tm.tm_sec % 5 == 0:
            grade, point_baston = strike_grade(joints, bboxList, key)

    if point_baston > -1:
        if shwSkltn:
            # Draw line for the baston
            frame = cv2.line(frame, joints[22], bboxList[point_baston], (70, 92, 105), 9) if (
                    22 in joints and bboxList[point_baston][0] >= 0 and bboxList[point_baston][1] >= 0) else frame
            # Draw the end point of the baston from the wrist
            frame = cv2.circle(frame, bboxList[point_baston], 10, (0, 0, 255), -1)

    fnt = cv2.FONT_HERSHEY_DUPLEX
    lab_sz = cv2.getTextSize(label, fnt, 1.2, 2)[0]
    labX = int(lab_sz[0])
    labY = int(lab_sz[1])

    frame = cv2.flip(frame, 1)
    if s_angle:
        joints_angles = joint_angles(joints, [(-1, -1), (-1, -1), (-1, -1), (-1, -1)])
        frame = angle_vis(frame, joints, joints_angles, bboxList)
    frame = cv2.rectangle(frame, (0, h), (labX, h - labY - 20), (255, 255, 255), cv2.FILLED)
    frame = cv2.putText(frame, label, (2, h - 15), fnt, 1.2, color_text, 2, cv2.LINE_AA)

    if grading:
        return frame, grade
    else:
        return frame, label


def angle_det(frame):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    results = pose.process(imgRGB)

    # print(results.pose_landmarks)
    joints = {}
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for jt_id, lm in enumerate(results.pose_landmarks.landmark):
            # print(jt_id, lm)
            cx, cy, thr = int(lm.x * w), int(lm.y * h), lm.visibility

            if thr > 0.5:
                joints[jt_id] = (cx, cy)
                print(jt_id, cx, cy, 'th: ', thr)

    joints_angles = joint_angles(joints, [(-1, -1), (-1, -1), (-1, -1), (-1, -1)])
    frame = angle_vis(frame, joints, joints_angles)

    return frame, joints_angles


def angle_vis(frame, joints, joints_angles, bboxList):
    h, w, c = frame.shape
    pt1, pt2, pt3, pt4 = bboxList

    # Display baston points
    if pt1:
        x, y = pt1
        x = w - x - 1
        frame = cv2.putText(frame, str(1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if pt2:
        x, y = pt2
        x = w - x - 1
        frame = cv2.putText(frame, str(2), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if pt3:
        x, y = pt3
        x = w - x - 1
        frame = cv2.putText(frame, str(3), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if pt4:
        x, y = pt4
        x = w - x - 1
        frame = cv2.putText(frame, str(4), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    # Show joints

    if 14 in joints:
        x, y = joints[14]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['right elbow']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if 12 in joints:
        x, y = joints[12]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['right shoulder']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if 24 in joints:
        x, y = joints[24]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['right hip']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if 26 in joints:
        x, y = joints[26]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['right knee']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if 13 in joints:
        x, y = joints[13]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['left elbow']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if 11 in joints:
        x, y = joints[11]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['left shoulder']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if 23 in joints:
        x, y = joints[23]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['left hip']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    if 25 in joints:
        x, y = joints[25]
        x = w - x - 1
        frame = cv2.putText(frame, str(joints_angles['left knee']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (246, 255, 0), 2, cv2.LINE_AA)

    return frame


def det_baston(frame, model):
    category_index = label_map_util.create_category_index_from_labelmap('training_v2/label_map.pbtxt')
    image_np = np.array(frame)
    height, width, _ = frame.shape

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    # start = time.time()
    detections = detect_fn(input_tensor, model)
    # print('first detection time:', time.time() - start)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=.4,
        agnostic_mode=False,
        skip_scores=True,
        skip_labels=True,
        skip_boxes=False)

    det_scrs = list(detections['detection_scores'])
    det_boxes = list(detections['detection_boxes'])

    hgh_scr = []

    for i in range(len(det_scrs)):
        if det_scrs[i] > 0.4:
            hgh_scr.append(det_scrs.index(det_scrs[i]))

    ymin, xmin, ymax, xmax = -1, -1, -1, -1
    for i in hgh_scr:
        # print("score", det_scrs[i])
        # print("boxes", det_boxes[i], "\n")
        ymin, xmin, ymax, xmax = det_boxes[i]

    pt1 = (int(xmin * width), int(ymin * height))
    pt2 = (int(xmin * width), int(ymax * height))
    pt3 = (int(xmax * width), int(ymax * height))
    pt4 = (int(xmax * width), int(ymin * height))
    # image_np_with_detections = cv2.circle(image_np_with_detections, pt1, 10, (0, 0, 255), -1)

    bboxList = [pt1, pt2, pt3, pt4]

    return image_np_with_detections, bboxList
