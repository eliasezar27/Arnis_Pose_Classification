import mediapipe as mp
from strikes import strike, joint_angles
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import cv2
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def pose_det(frame, model):
    # Uncomment flip to extract angles, visually
    # frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame, bboxList = det_baston(frame, model)
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
                # print(jt_id, cx, cy, 'th: ', thr)

    # joints_angles = joint_angles(joints, [(-1,-1),(-1,-1),(-1,-1),(-1,-1)])
    # frame = angle_vis(frame, joints, joints_angles)

    label, point_baston = strike(joints, bboxList)
    color_text = (255, 0, 0) if 'Block' in label else (0, 255, 0)

    frame = cv2.line(frame, joints[22], bboxList[point_baston], (70, 92, 105), 9) if (22 in joints and bboxList[point_baston][0] >= 0 and bboxList[point_baston][1] >= 0) else frame
    frame = cv2.circle(frame, bboxList[point_baston], 10, (0, 0, 255), -1)

    lab_len = int(len(label) * 26.5)
    if lab_len > 450:
        lab_len = 480
    frame = cv2.flip(frame, 1)
    frame = cv2.rectangle(frame, (0, h), (lab_len, h-45), (255, 255, 255), cv2.FILLED)
    frame = cv2.putText(frame, label, (1, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_text, 2, cv2.LINE_AA)

    return frame


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


def angle_vis(frame, joints, joints_angles):

    if 14 in joints:
        frame = cv2.putText(frame, str(joints_angles['right elbow']), joints[14], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    if 12 in joints:
        frame = cv2.putText(frame, str(joints_angles['right shoulder']), joints[12], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    if 24 in joints:
        frame = cv2.putText(frame, str(joints_angles['right hip']), joints[24], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    if 26 in joints:
        frame = cv2.putText(frame, str(joints_angles['right knee']), joints[26], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    if 13 in joints:
        frame = cv2.putText(frame, str(joints_angles['left elbow']), joints[13], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    if 11 in joints:
        frame = cv2.putText(frame, str(joints_angles['left shoulder']), joints[11], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    if 23 in joints:
        frame = cv2.putText(frame, str(joints_angles['left hip']), joints[23], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    if 25 in joints:
        frame = cv2.putText(frame, str(joints_angles['left knee']), joints[25], cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 255, 0), 2, cv2.LINE_AA)

    return frame


def det_baston(frame, model):
    category_index = label_map_util.create_category_index_from_labelmap('training_v2/label_map.pbtxt')
    image_np = np.array(frame)
    height, width, _ = frame.shape

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, model)

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
        skip_boxes=True)

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