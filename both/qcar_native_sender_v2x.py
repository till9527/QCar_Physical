import os
import signal
import numpy as np
import threading
from threading import Thread
import time
import socket
import struct
import math
import select
import cv2

# --- Quanser Imports ---
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from pal.utilities.vision import Camera2D

# V2X Import
from pal.products.traffic_light import TrafficLight

# ================= CONFIGURATION =================
# --- Network (PC Connection) ---
COMPUTER_IP = "192.168.2.16"  # <--- CONFIRM THIS IS YOUR PC IP
PORT = 8080

# --- Camera ---
CAMERA_ID = "3"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30

# --- V2X / Traffic Lights ---
TRAFFIC_LIGHTS_CONFIG = [
    {"id": 1, "ip": "192.168.2.15", "location": [2.113, 0.204], "yaw_deg": 0},
    {"id": 2, "ip": "192.168.2.14", "location": [-1.909, 0.738], "yaw_deg": 180},
]

# --- Controller ---
tf = 600
startDelay = 1
controllerUpdateRate = 100
v_cruise = 0.5  # Target cruising speed
K_p = 0.1
K_i = 1
enableSteeringControl = True
K_stanley = 1
nodeSequence = [10, 4, 20, 10]

# --- Global State ---
is_running = True
latest_frame = None
frame_lock = threading.Lock()

# Shared Control States
car_state = "GO"  # From PC (Perception)
car_state_lock = threading.Lock()

traffic_light_statuses = ["UNKNOWN"] * len(TRAFFIC_LIGHTS_CONFIG)  # From V2X


# ================= V2X HELPER FUNCTIONS =================
def generate_rotated_geofencing_areas(config_list):
    """Generates geofencing bounds based on traffic light config."""
    generated_areas = []
    # Box dimensions relative to light: 0.3m to 0.8m in front, 0.6m wide
    # Note: Using the specific offsets found in your Multi_V2X.py
    local_corner_1 = (0.3, -0.5)
    local_corner_2 = (-0.3, -1.0)

    for light in config_list:
        center_x = light["location"][0]
        center_y = light["location"][1]
        yaw_rad = math.radians(light["yaw_deg"])

        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Rotate corners
        def rotate(x, y):
            return (x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw)

        rx1, ry1 = rotate(local_corner_1[0], local_corner_1[1])
        rx2, ry2 = rotate(local_corner_2[0], local_corner_2[1])

        world_c1 = (center_x + rx1, center_y + ry1)
        world_c2 = (center_x + rx2, center_y + ry2)

        x_min, x_max = min(world_c1[0], world_c2[0]), max(world_c1[0], world_c2[0])
        y_min, y_max = min(world_c1[1], world_c2[1]), max(world_c1[1], world_c2[1])

        generated_areas.append(
            {
                "name": f"Traffic Light {light['id']}",
                "bounds": [(x_min, y_min), (x_max, y_max)],
            }
        )
    return generated_areas


def is_inside_geofence(position, geofence):
    (x_min, y_min), (x_max, y_max) = geofence
    return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max


# Initialize Geofences and Lights
geofencing_areas = generate_rotated_geofencing_areas(TRAFFIC_LIGHTS_CONFIG)
traffic_lights_objects = [TrafficLight(cfg["ip"]) for cfg in TRAFFIC_LIGHTS_CONFIG]

# ================= THREADS =================


def traffic_lights_status_thread():
    """Continuously fetches status from traffic lights over Wi-Fi."""
    global traffic_light_statuses
    status_map = {"1": "RED", "2": "YELLOW", "3": "GREEN"}

    while is_running:
        new_statuses = []
        for light in traffic_lights_objects:
            try:
                s = light.status()
                new_statuses.append(status_map.get(s, "UNKNOWN"))
            except:
                new_statuses.append("UNKNOWN")
        traffic_light_statuses = new_statuses
        time.sleep(0.5)


def receiver_thread_func(sock):
    """Receives STOP/GO commands from the PC (Perception)."""
    global is_running, car_state
    print("Receiver thread started...")
    while is_running:
        readable, _, _ = select.select([sock], [], [], 1.0)
        if sock in readable:
            try:
                data = sock.recv(1024)
                if data:
                    command = data.decode("utf-8").strip()
                    # We accept the last valid command
                    if "STOP" in command:
                        cmd = "STOP"
                    elif "GO" in command:
                        cmd = "GO"
                    else:
                        cmd = None

                    if cmd:
                        with car_state_lock:
                            car_state = cmd
                else:
                    break
            except:
                break
    print("Receiver thread stopped.")


def camera_thread_func(camera):
    """Reads camera frames."""
    global latest_frame, is_running
    while is_running:
        if camera.read():
            with frame_lock:
                latest_frame = camera.imageData
        time.sleep(1 / FRAME_RATE)


# ================= CONTROLLERS =================


class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )


class SteeringController:
    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]
        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except:
            return 0
        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p - wp_1, v_uv)
        if s >= v_mag and (self.cyclic or self.wpi < self.N - 2):
            self.wpi += 1
        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)
        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


# ================= MAIN CONTROL LOOP =================


def control_thread_func(initialPose, waypointSequence, calibrationPose, calibrate):
    global is_running, car_state

    speedController = SpeedController(kp=K_p, ki=K_i)
    steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf = QCarEKF(x_0=initialPose)
    gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)

    print("Control Loop Started.")
    with qcar, gps:
        t0 = time.time()
        t = 0
        delta = 0

        while (t < tf + startDelay) and is_running:
            tp = t
            t = time.time() - t0
            dt = t - tp

            # 1. Read Sensors
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]
            p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach

            # 2. Determine Speed Command
            target_speed = 0.0

            if t > startDelay:
                # A. Check Perception Command (PC)
                perception_says_go = False
                with car_state_lock:
                    if car_state == "GO":
                        perception_says_go = True

                # B. Check V2X Command (Traffic Lights)
                v2x_stop_override = False
                current_pos = (x, y)

                for i, geofence in enumerate(geofencing_areas):
                    if is_inside_geofence(current_pos, geofence["bounds"]):
                        # We are at a light, check its status
                        status = traffic_light_statuses[i]
                        if status == "RED":
                            v2x_stop_override = True
                            # Optional: Print only once to avoid spam
                            # print(f"V2X Stop: {geofence['name']} is RED")

                # C. Final Logic: Go only if PC says GO *AND* V2X doesn't override
                if perception_says_go and not v2x_stop_override:
                    target_speed = v_cruise
                else:
                    target_speed = 0.0

                # D. Update Controllers
                u = speedController.update(v, target_speed, dt)
                delta = steeringController.update(p, th, v)
                qcar.write(u, delta)
            else:
                qcar.write(0, 0)

    qcar.write(0, 0)
    is_running = False


# ================= EXECUTION =================

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *args: globals().update(is_running=False))

    # Setup Map / GPS
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()

    # Input for calibration
    calibrate = "y" in input("Do you want to recalibrate? (y/n): ")
    calibrationPose = [0, 2, -np.pi / 2]

    # Initialize Threads
    try:
        # Connect to PC
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to PC at {COMPUTER_IP}:{PORT}...")
        client_socket.connect((COMPUTER_IP, PORT))
        print("Connected.")

        camera = Camera2D(
            cameraId=CAMERA_ID,
            frameWidth=IMAGE_WIDTH,
            frameHeight=IMAGE_HEIGHT,
            frameRate=FRAME_RATE,
        )

        # Start Helper Threads
        t_cam = Thread(target=camera_thread_func, args=(camera,))
        t_net = Thread(target=receiver_thread_func, args=(client_socket,))
        t_v2x = Thread(target=traffic_lights_status_thread)
        t_ctrl = Thread(
            target=control_thread_func,
            args=(initialPose, waypointSequence, calibrationPose, calibrate),
        )

        t_cam.start()
        t_net.start()
        t_v2x.start()
        t_ctrl.start()

        # Main Loop: Send Video
        while is_running:
            local_frame = None
            with frame_lock:
                if latest_frame is not None:
                    local_frame = np.ascontiguousarray(latest_frame)

            if local_frame is not None:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, encoded_img = cv2.imencode(".jpg", local_frame, encode_param)
                data = np.array(encoded_img).tobytes()

                # Send size then data
                try:
                    client_socket.sendall(struct.pack(">L", len(data)))
                    client_socket.sendall(data)
                except:
                    is_running = False

            time.sleep(0.02)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        is_running = False
        print("Shutting down...")
        time.sleep(1)
        if "camera" in locals():
            camera.terminate()
        if "client_socket" in locals():
            client_socket.close()
