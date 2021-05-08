import cv2
from pose_est import pose_det
import numpy as np
import time

# Object Detection Modules
import tensorflow as tf
from object_detection.utils import config_util
import os
from object_detection.builders import model_builder
import cv2
CONFIG_PATH = "training_v2/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

strt = time.time()
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('training_v2', 'ckpt-10')).expect_partial()
load_model = time.time()-strt

print("loading time: ",load_model)

# cap = cv2.VideoCapture('Arnis_12 Striking Techniques.mp4')
cap = cv2.VideoCapture('vids/Arnis_12 Striking Techniques.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        new_image = np.zeros(frame.shape, frame.dtype)

        alpha = 1.3 # contrast 1.0 - 3.0
        beta = 66 # brightness 0 - 100

        new_image[:, :, :] = np.clip(alpha * frame[:, :, :] + beta, 0, 255)

        # Display the resulting frame
        # frame = pose_det(frame)
        new_image = pose_det(new_image, detection_model)
        # frame = cv2.flip(frame, 1)
        cv2.imshow('Frame', new_image)
        # cv2.imshow('Pred', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
