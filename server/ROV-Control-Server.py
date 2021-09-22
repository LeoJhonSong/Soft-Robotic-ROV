#! /usr/bin/env python3

import argparse
import logging
import socket
import threading
import time

import cv2
import numpy as np
import yaml
from flask import Flask, Response, render_template, request

from controller.auv import Auv
from controller.utils import tprint
from CppPythonSocket import Server

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__, static_folder='static', template_folder='templates')
# filter Werkzeug logger to error level
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def video_stream():
    # grab global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock
    if args['source'] == 'socket':
        server = Server("127.0.0.1", 8010)
    elif args['source'] in ['zed', 'camera']:
        cap = cv2.VideoCapture(0)
    while True:
        if args['source'] == 'socket':
            frame = server.receive_image()
        elif args['source'] == 'zed':
            _, frame = cap.read()
            frame = frame[:, :frame.shape[1] // 2:, :]
        elif args['source'] == 'camera':
            _, frame = cap.read()
        else:
            frame = np.random.randint(255, size=(900, 800, 3), dtype=np.uint8)
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
            # check if the output frame is available, otherwise skip the iteration of the loop
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


@app.route("/tune", methods=['GET', 'POST'])
def form():
    p_dict = {i: 0 for i in range(10)}
    if request.method == 'POST':
        for i in range(7):
            p_dict[i] = float(request.form.get('chamber' + str(i + 1)))
        p_dict[9] = float(request.form.get('hand'))
        robot.arm.len2pressures(robot.arm.inverse_kinematics(0, 0, 0), pressures_dict=p_dict)
    return render_template("tune.html", p_dict=p_dict)


@app.route("/video_feed")
def video_feed():
    """return the response generated along with the specific media type (mime type), a byte array of a JPEG image
    """
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/auv/joystick", methods=['GET', 'POST'])
def js_auv():
    velocity = request.json
    # tprint(f'Vx: {velocity["vx"]:.03f}, Vy: {velocity["vy"]:.03f}')
    robot.set_move((velocity['vx'], velocity['vy'], velocity['steer'], velocity['vz']))
    return velocity


@app.route("/arm/release", methods=['GET', 'POST'])
def release_arm():
    robot.arm.release()
    return {}


@app.route("/led", methods=['GET', 'POST'])
def led():
    data = request.json
    robot.set_led(int(data['led']))
    return {}


@app.route("/arm/reset", methods=['GET', 'POST'])
def reset_arm():
    robot.arm.reset()
    return {}


@app.route("/arm/collect", methods=['GET', 'POST'])
def collect():
    robot.get_sensors_data()
    robot.arm.collect()
    return {}


@app.route("/arm/fold", methods=['GET', 'POST'])
def fold():
    data = request.json
    robot.get_sensors_data()
    robot.arm.fold(data['state'])
    return {}


@app.route("/arm/joystick", methods=['GET', 'POST'])
def js_arm():
    data = request.json
    scale = 1000
    x = scale * float(data['x'])
    y = scale * float(data['y'])
    elg = 40 * float(data['elg'])
    tprint(f'💪 📍 Arm x: {x:.03f}mm, y: {y:.03f}mm, elongation: {elg:.03f}mm (manual)')
    p_dict = {i: elg for i in range(6, 9)}
    time.sleep(0.1)
    robot.get_sensors_data()
    robot.arm.len2pressures(robot.arm.inverse_kinematics(x, y, 0), pressures_dict=p_dict)
    robot.arm.hand(data['hand'])
    return data


is_auto = False


@app.route("/auto", methods=['GET', 'POST'])
def auto_trigger():
    global is_auto
    data = request.json
    is_auto = data['auto']
    if is_auto:
        robot.state = 'land'
        tprint('🤖 switch to auto mode')
        auto()
    else:
        robot.reset()
        tprint('🧠 switch to manual mode')
    return {}


def auto():
    global is_auto
    quit_flag = False
    robot.arm.reset()
    robot.set_led(1)
    while True:
        if not is_auto:
            break
        # 更新target, arm数据
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as visual_socket:
            try:
                visual_socket.connect(('127.0.0.1', 8080))
            except ConnectionRefusedError:
                print('[Visual Info Client] lost connection')
                continue
            # send flags to visual server
            # threads_quit_flag: 2
            visual_socket.send(bytes(str(quit_flag * 2).encode()))
            # receive data from ROV then update target and arm
            visual_info_dict = yaml.load(visual_socket.recv(1024), Loader=yaml.Loader)
        # quit_flag置1后待运行到下一循环, 将quit_flag发送给visual_server后再break
        # update AUV data
        robot.target.update(visual_info_dict["target"])
        robot.visual_arm.update(visual_info_dict["arm"])
        robot.get_sensors_data()  # 主要是获取深度
        # time.sleep(0.001)  # 1ms
        # 状态机状态跳转并给出抓取判断
        grasp_state = robot.state_machine()
        # 软体臂控制
        if grasp_state == 'ready':
            robot.visual_arm.arm_is_working = True
            robot.arm.release()
            robot.arm.hand('open')
            time.sleep(5)
            tc = robot.target.center
            robot.visual_arm.start_time = time.time()
            tprint('💪 Arm ready')
        elif grasp_state == 'activated':
            if robot.arm.reached in ['not', 'wait']:
                robot.arm.controller.send((tc + (robot.arm.initZ,), robot.visual_arm.marker_position + (robot.arm.initZ,)))
            elif robot.arm.reached == 'yes':
                # 到达位置后伸长手臂
                robot.arm.pressures[6:9] = [40] * 3
                robot.arm.set_Pressures()
                time.sleep(2)
                # 抓取
                robot.arm.hand('close')
                time.sleep(2)
                # 收集
                robot.arm.collect()
                robot.grasp_state = 'idle'
                robot.visual_arm.arm_is_working = False
            elif robot.arm.reached == 'out':
                robot.arm.reset()
                robot.grasp_state = 'idle'
                robot.visual_arm.arm_is_working = False
        elif grasp_state == 'idle':
            robot.visual_arm.arm_is_working = False


with Auv() as robot:
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='0.0.0.0', help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=8000, help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-s", "--source", type=str, default='socket', help="video source (socket or zed)")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=video_stream, daemon=True)
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=False, threaded=True, use_reloader=False)
