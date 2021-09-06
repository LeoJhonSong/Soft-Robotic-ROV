# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request
import threading
import argparse
import cv2
from CppPythonSocket import Server

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__, static_folder='static', template_folder='templates')
# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(src=0).start()


def video_stream():
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock
    server = Server("127.0.0.1", 8000)
    while True:
        # frame = vs.read()
        frame = server.receive_image()
        # frame = imutils.resize(frame, width=400)
        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = frame.copy()


def generate():
    """generator used to encode outputFrame as JPEG data
    """
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
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """return the response generated along with the specific media type (mime type), a byte array of a JPEG image
    """
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/joysticks/rov", methods=['GET', 'POST'])
def js_rov():
    data = request.json
    print(data)
    return data


@app.route("/joysticks/arm", methods=['GET', 'POST'])
def js_arm():
    data = request.json
    print(data)
    return data


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='0.0.0.0', help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=8000, help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=video_stream, daemon=True)
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

# release the video stream pointer
# vs.stop()
