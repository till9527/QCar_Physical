import os
import signal
import numpy as np
import threading
from threading import Thread
import time
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from pal.products.traffic_light import TrafficLight
import math
import socket
import struct
import select
import cv2  # OpenCV for compression
from pal.utilities.vision import Camera2D  # Helper for QCar Cameras

# ================ Experiment Configuration ================
# ===== Timing Parameters
tf = 600
startDelay = 1
controllerUpdateRate = 100
# ================ Camera & Streaming Config ================
COMPUTER_IP = "192.168.2.16"  # <--- CHANGE THIS to your PC's IP address
PORT = 8080
CAMERA_ID = "3"  # "3" is usually the CSI Rear/Top camera
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30
is_running = True
# Global Shared Resources for Camera
latest_frame = None
frame_lock = threading.Lock()

# Shared command state from computer receiver
car_state = "GO"
car_state_lock = threading.Lock()
last_force_cmd_time = 0.0
FORCE_CMD_TIMEOUT_S = 1.0
# ===== Speed Controller Parameters
v_cruise = 0.5
K_p = 0.1
K_i = 1

# ===== Steering Controller Parameters
enableSteeringControl = True
K_stanley = 1
nodeSequence = [10, 4, 20, 10]

# region : Initial setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    import qlabs_setup

    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]],
    )
    calibrate = False
else:
    calibrate = "y" in input("do you want to recalibrate?(y/n)")


# Define the calibration pose
# Calibration pose is either [0,0,-pi/2] or [0,2,-pi/2]
# Comment out the one that is not used
# calibrationPose = [0,0,-np.pi/2]
calibrationPose = [0, 2, -np.pi / 2]

# Stop Configuration
deceleration_rate = 0.1  # Rate of deceleration (m/s^2)
acceleration_rate = 0.1  # Rate of acceleration (m/s^2)
stop_tolerance = 0.2  # Stop tolerance (meters)
stop_distance_offset = 0.5  # Distance to stop before the target

# Ensure math is imported at the top

# 1. Define Traffic Light Configuration (IDs, IPs, and Rotations)
# We add rotation here so the geofence knows which way the light is facing
TRAFFIC_LIGHTS_CONFIG = [
    {"id": 1, "ip": "192.168.2.15", "location": [2.113, 0.204], "yaw_deg": 0},
    {"id": 2, "ip": "192.168.2.14", "location": [-1.909, 0.738], "yaw_deg": 180},
]

# 2. Initialize Traffic Lights using the config
traffic_lights = [TrafficLight(cfg["ip"]) for cfg in TRAFFIC_LIGHTS_CONFIG]
traffic_light_statuses = ["UNKNOWN"] * len(TRAFFIC_LIGHTS_CONFIG)
traffic_light_status_times = [0.0] * len(TRAFFIC_LIGHTS_CONFIG)
traffic_light_lock = threading.Lock()
TRAFFIC_LIGHT_POLL_INTERVAL_S = 0.1
TRAFFIC_LIGHT_STALE_THRESHOLD_S = 1.0


def generate_rotated_geofencing_areas(config_list):
    """
    Generates geofencing areas by applying rotation to local bounds.
    SCALING_FACTOR (0.1) is manually applied here to the coordinates.
    """
    generated_areas = []

    # --- UPDATED COORDINATES ---
    # Original: (1.0, -9.0) and (-2.0, -13.0)
    # Applied 0.1 scaling: (0.1, -0.9) and (-0.2, -1.3)
    #
    # We also widen the X range slightly (from 0.3m to 0.6m)
    # to catch the car even if GPS drifts slightly sideways.

    # Box: 0.9m to 1.5m in front of the light, 0.6m wide centered-ish
    local_corner_1 = (0.3, -0.3)
    local_corner_2 = (-0.3, -0.8)

    for light in config_list:
        center_x = light["location"][0]
        center_y = light["location"][1]
        yaw_rad = math.radians(light["yaw_deg"])

        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Rotate local_corner_1
        rot_x1 = local_corner_1[0] * cos_yaw - local_corner_1[1] * sin_yaw
        rot_y1 = local_corner_1[0] * sin_yaw + local_corner_1[1] * cos_yaw
        world_c1 = (center_x + rot_x1, center_y + rot_y1)

        # Rotate local_corner_2
        rot_x2 = local_corner_2[0] * cos_yaw - local_corner_2[1] * sin_yaw
        rot_y2 = local_corner_2[0] * sin_yaw + local_corner_2[1] * cos_yaw
        world_c2 = (center_x + rot_x2, center_y + rot_y2)

        # Create AABB
        x_min, x_max = min(world_c1[0], world_c2[0]), max(world_c1[0], world_c2[0])
        y_min, y_max = min(world_c1[1], world_c2[1]), max(world_c1[1], world_c2[1])

        # --- DEBUG PRINT ---
        # This will verify the box is actually on your map (e.g., values < 3.0m)
        print(
            f"Created Geofence for Light {light['id']}: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]"
        )

        generated_areas.append(
            {
                "name": f"Traffic Light {light['id']}",
                "bounds": [(x_min, y_min), (x_max, y_max)],
            }
        )

    return generated_areas


# 3. Initialize the areas
geofencing_areas = generate_rotated_geofencing_areas(TRAFFIC_LIGHTS_CONFIG)


def is_inside_geofence(position, geofence):
    (x_min, y_min), (x_max, y_max) = geofence
    return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max


def is_inside_geofence_padded(position, geofence, padding=0.0):
    (x_min, y_min), (x_max, y_max) = geofence
    return (x_min - padding) <= position[0] <= (x_max + padding) and (
        y_min - padding
    ) <= position[1] <= (y_max + padding)


# Used to enable safe keyboard triggered shutdown
global KILL_THREAD, v_ref
KILL_THREAD = False
v_ref = v_cruise


def sig_handler(*args):
    global KILL_THREAD, is_running
    KILL_THREAD = True
    is_running = False


signal.signal(signal.SIGINT, sig_handler)


class SpeedController:

    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki

        self.ei = 0

    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e

        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )

        return 0


class SteeringController:

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

    # ==============  SECTION B -  Steering Control  ====================
    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p - wp_1, v_uv)

        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)

        self.p_ref = ep
        self.th_ref = tangent

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


# def check_geofencing_and_stop(position):
#     global v_ref, has_stopped_at

#     for i, geofence in enumerate(geofencing_areas):
#         name = geofence["name"]
#         inside = is_inside_geofence(position, geofence["bounds"])
#         traffic_light_status = traffic_light_statuses[i]  # Get respective traffic light status
#         if inside:
#             print(f"Entered geofencing area of {name}")

#             if traffic_light_status == "RED" and not has_stopped_at[name]:
#                 v_ref = 0
#                 has_stopped_at[name] = True
#                 print(f" Stopping at {name} due to RED light!")

#             elif traffic_light_status == "GREEN" and has_stopped_at[name]:
#                 v_ref = v_cruise
#                 has_stopped_at[name] = False
#                 print(f" Traffic light at {name} turned GREEN. Resuming movement.")
#         if not inside and has_stopped_at[name]:
#             has_stopped_at[name] = False


def traffic_lights_status_thread():
    global traffic_light_statuses, traffic_light_status_times
    status_map = {"1": "RED", "2": "YELLOW", "3": "GREEN"}
    while is_running and (not KILL_THREAD):
        for i, light in enumerate(traffic_lights):
            try:
                s = light.status()
                mapped_status = status_map.get(s, "UNKNOWN")
            except Exception:
                mapped_status = "UNKNOWN"

            now = time.time()
            with traffic_light_lock:
                traffic_light_statuses[i] = mapped_status
                traffic_light_status_times[i] = now

        time.sleep(TRAFFIC_LIGHT_POLL_INTERVAL_S)


def receiver_thread_func(sock):
    """Receives STOP/GO/FORCE commands from the computer receiver."""
    global is_running, car_state, KILL_THREAD, last_force_cmd_time
    print("Receiver thread started...")
    while is_running and (not KILL_THREAD):
        readable, _, _ = select.select([sock], [], [], 1.0)
        if sock in readable:
            try:
                data = sock.recv(1024)
                if data:
                    command = data.decode("utf-8").strip()
                    if "FORCE_STOP" in command:
                        cmd = "FORCE_STOP"
                    elif "FORCE_GO" in command:
                        cmd = "FORCE_GO"
                    elif "STOP" in command:
                        cmd = "STOP"
                    elif "GO" in command:
                        cmd = "GO"
                    else:
                        cmd = None

                    if cmd:
                        with car_state_lock:
                            car_state = cmd
                            if cmd in ("FORCE_GO", "FORCE_STOP"):
                                last_force_cmd_time = time.time()
                else:
                    is_running = False
                    KILL_THREAD = True
                    break
            except Exception:
                is_running = False
                KILL_THREAD = True
                break
    print("Receiver thread stopped.")


def controlLoop():
    # region controlLoop setup
    global KILL_THREAD, is_running, car_state
    u = 0
    delta = 0
    countMax = controllerUpdateRate / 10
    count = 0
    geofence_padding_m = 0.15
    last_geofence_name = None
    last_geofence_log_time = 0
    # endregion

    # region Controller initialization
    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)
    # endregion

    # region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        gps = memoryview(b"")
    # endregion
    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD) and is_running:
            # region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t - tp
            # endregion

            # region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                gps_position = None
                if gps.readGPS():
                    gps_position = (gps.position[0], gps.position[1])
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach
            # endregion

            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                with car_state_lock:
                    state = car_state
                    if (
                        state in ("FORCE_GO", "FORCE_STOP")
                        and (time.time() - last_force_cmd_time) > FORCE_CMD_TIMEOUT_S
                    ):
                        car_state = "GO"
                        state = "GO"
                        print("Force mode expired -> AUTO")

                v2x_stop_override = False
                current_pos = gps_position if gps_position is not None else (x, y)
                current_geofence_name = None
                current_geofence_status = "UNKNOWN"

                for i, geofence in enumerate(geofencing_areas):
                    if is_inside_geofence_padded(
                        current_pos, geofence["bounds"], geofence_padding_m
                    ):
                        current_geofence_name = geofence["name"]
                        with traffic_light_lock:
                            status = traffic_light_statuses[i]
                            status_age = time.time() - traffic_light_status_times[i]

                        if status_age > TRAFFIC_LIGHT_STALE_THRESHOLD_S:
                            status = "UNKNOWN"

                        current_geofence_status = status
                        if status == "RED":
                            v2x_stop_override = True

                now = time.time()
                if current_geofence_name is not None:
                    if (
                        current_geofence_name != last_geofence_name
                        or (now - last_geofence_log_time) > 1.0
                    ):
                        print(
                            f"Inside {current_geofence_name} | pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}) | light={current_geofence_status}"
                        )
                        last_geofence_log_time = now
                    last_geofence_name = current_geofence_name
                else:
                    last_geofence_name = None

                if state == "FORCE_GO":
                    target_speed = v_cruise
                elif state == "FORCE_STOP":
                    target_speed = 0.0
                else:
                    perception_says_go = state == "GO"
                    if perception_says_go and not v2x_stop_override:
                        target_speed = v_cruise
                    else:
                        target_speed = 0.0

                u = speedController.update(v, target_speed, dt)

                # region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0
                # endregion

            qcar.write(u, delta)
            # endregion

            count += 1
            if count >= countMax and t > startDelay:
                count = 0
                continue

    qcar.write(0, 0)
    is_running = False


# endregion
def camera_thread_func(camera):
    """Continuously reads from the camera into a global variable."""
    global latest_frame, is_running
    print("Camera thread started...")
    while is_running:
        if camera.read():
            with frame_lock:
                latest_frame = camera.imageData
        time.sleep(1 / FRAME_RATE)
    print("Camera thread stopped.")


# region : Setup and run experiment
if __name__ == "__main__":
    # 1. Start existing threads
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to computer at {COMPUTER_IP}:{PORT}...")
    client_socket.connect((COMPUTER_IP, PORT))
    print("Connection established.")

    # 2. Initialize Camera
    camera = Camera2D(
        cameraId=CAMERA_ID,
        frameWidth=IMAGE_WIDTH,
        frameHeight=IMAGE_HEIGHT,
        frameRate=FRAME_RATE,
    )

    trafficLightsThread = Thread(target=traffic_lights_status_thread)
    trafficLightsThread.start()

    receiverThread = Thread(target=receiver_thread_func, args=(client_socket,))
    receiverThread.start()

    controlThread = Thread(target=controlLoop)
    controlThread.start()

    # 2. Start NEW Camera threads
    camThread = Thread(target=camera_thread_func, args=(camera,))
    camThread.start()

    try:
        while is_running and (not KILL_THREAD):
            local_frame = None
            with frame_lock:
                if latest_frame is not None:
                    local_frame = np.ascontiguousarray(latest_frame)

            if local_frame is not None:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, encoded_img = cv2.imencode(".jpg", local_frame, encode_param)

                data = np.array(encoded_img)
                frame_bytes = data.tobytes()

                message_header = struct.pack(">L", len(frame_bytes))
                client_socket.sendall(message_header)
                client_socket.sendall(frame_bytes)

            time.sleep(0.02)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        KILL_THREAD = True
        is_running = False
        trafficLightsThread.join()
        controlThread.join()
        camThread.join()
        receiverThread.join()
        if camera:
            camera.terminate()
        if client_socket:
            client_socket.close()
        print("All threads stopped.")
# endregion
