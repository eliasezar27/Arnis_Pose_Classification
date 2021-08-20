from flask import Flask, render_template, Response, request
import imutils
import threading
from imutils.video import VideoStream
import time
from pose_est import pose_det

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
# Load restored checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('training_v2', 'ckpt-10')).expect_partial()
load_model = time.time()-strt

print("obj det model loading time: ", load_model)


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

vs = VideoStream(src=1).start()
# time.sleep(2.0)
prev_frame_time = 0

# Pose key start ADDED
pose_key = 1


# Index page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# Pose Classification Page
@app.route('/opencam', methods=['GET', 'POST'])
def index2():
    answer = request.form['response']
    return render_template('index.html', ans=answer)


# Read poses from camera input
def camera():
    global vs, outputFrame, lock, prev_frame_time, pose_key
    # grab global references to the video stream, output frame, and
    # lock variables

    while True:
        # read the next frame from the video stream, resize it,
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # ADDED: get frame dimension
        h, w, c = frame.shape

        # font color for grade
        clr_grd = (0, 0, 255)

        start = time.time()
        # added arg: pose key, added var: grade
        frame, grade = pose_det(frame, detection_model, pose_key)
        print('Pose classification prediction time: ', time.time() - start)

        # Threshold ADDED
        # next pose when threshold is greater than 74
        if grade >= 90:
            clr_grd = (0, 255, 0)

            # next pose key until 23rd pose
            if pose_key < 26:
                pose_key = pose_key + 1
            else:
                pose_key = 1

        # Visualize grade
        frame = cv2.rectangle(frame, (w, 0), (w - 275, 45), (255, 255, 255), cv2.FILLED)
        frame = cv2.putText(frame, "Grade:" + str(grade), (w - 275, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, clr_grd, 2,
                            cv2.LINE_AA)
        print(str(pose_key) + " " + str(grade))

        # Save prediction/classification time in text file
        with open('speed.txt', 'a') as f:
            f.write(str(time.time() - start) + ", ")

        # FPS
        new_frame_time = time.time()
        fps = int(1/(new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        fps = str(fps)
        frame = cv2.putText(frame, "FPS: " + fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()


# Generate video output in the webpage
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    t = threading.Thread(target=camera)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(debug=True, threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
