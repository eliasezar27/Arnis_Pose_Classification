from flask import Flask, render_template, Response, request
import imutils
import threading
from imutils.video import VideoStream
import time
from pose_est import pose_det
from statistics import mean

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

# Store grades
prev_grade = 0
grade = 0
ave_grade = [1 for i in range(27)]


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
    global vs, outputFrame, lock, prev_frame_time, pose_key, grade, prev_grade, ave_grade
    blur_end = False
    fnt = cv2.FONT_HERSHEY_DUPLEX
    # grab global references to the video stream, output frame, and
    # lock variables

    while True:
        # print('Grade: ', grade, ' Prev grade: ', prev_grade)
        # Pause 2 sec if pose passed
        if prev_grade >= 75:
            time.sleep(2.0)

        # read the next frame from the video stream, resize it,
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # ADDED: get frame dimension
        h, w, c = frame.shape
        # print('Height: ', h, 'Width: ', w)

        # font color for grade
        clr_grd = (0, 0, 255)

        start = time.time()
        # added arg: pose key, added var: grade
        frame, grade = pose_det(frame, detection_model, pose_key)
        print('Pose classification prediction time: ', time.time() - start)

        # Compute overall grade
        ave_grade[pose_key] = grade

        # Threshold ADDED
        # next pose when threshold is greater than 74
        if grade >= 75:
            clr_grd = (0, 255, 0)

            # next pose key until 23rd pose
            if pose_key < 26:
                pose_key = pose_key + 1
            else:
                pose_key = 0
                blur_end = True

        # Visualize grade
        txt_grd = "Grade:" + str(grade)
        txt_grdsz = cv2.getTextSize(txt_grd, fnt, 1.2, 2)[0]

        frame = cv2.rectangle(frame, (w, 0), (w - txt_grdsz[0], txt_grdsz[1]), (255, 255, 255), cv2.FILLED)
        frame = cv2.putText(frame, txt_grd, (w - txt_grdsz[0], txt_grdsz[1]), fnt, 1.2, clr_grd, 2,cv2.LINE_AA)
        # print(str(pose_key) + " " + str(grade))

        # Save prediction/classification time in text file
        with open('speed.txt', 'a') as f:
            f.write(str(time.time() - start) + ", ")

        # FPS
        new_frame_time = time.time()
        fps = int(1/(new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        fps = str(fps)
        frame = cv2.putText(frame, "FPS: " + fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Check whether the last pose is done
        if blur_end:
            # Blur video feed after completing 24 needed poses
            overlay = frame.copy()
            output = frame.copy()
            overlay = cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

            # Put text after completing all poses
            txt1 = "YOU HAVE COMPLETED"
            txt2 = "THE 24 BASIC TECHNIQUES OF ARNIS"
            txt3 = "GRADE: " + str(round(mean(ave_grade[3:]), 2))
            text_sz1 = cv2.getTextSize(txt1, fnt, 1, 2)[0]
            text_sz2 = cv2.getTextSize(txt2, fnt, 1, 2)[0]
            text_sz3 = cv2.getTextSize(txt3, fnt, 2, 2)[0]

            txtX1 = int((output.shape[1] - text_sz1[0]) / 2)
            txtY1 = int(((output.shape[0] + text_sz1[1]) / 2) - (text_sz1[1] / 2))

            txtX2 = int((output.shape[1] - text_sz2[0]) / 2)
            txtY2 = int(((output.shape[0] + text_sz1[1]) / 2) + (text_sz1[1]))

            txtX3 = int((output.shape[1] - text_sz3[0]) / 2)
            txtY3 = int(((output.shape[0] + text_sz1[1]) / 4))

            frame = cv2.putText(frame, txt1, (txtX1, txtY1), fnt, 1, (255, 255, 255), 2)
            frame = cv2.putText(frame, txt2, (txtX2, txtY2), fnt, 1, (255, 255, 255), 2)
            frame = cv2.putText(frame, txt3, (txtX3, txtY3), fnt, 2, (0, 255, 0), 2)

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            prev_grade = grade
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
