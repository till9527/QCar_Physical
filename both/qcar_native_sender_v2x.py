# qcar_sender_with_control_and_v2x.py - Runs ON THE QCAR

import socket
import struct
import threading
import time
import select
import signal
import numpy as np


# --- Imports from vehicle_control_with_camera.py ---
from pal.products.qcar import (
    QCar,
    QCarGPS,
    IS_PHYSICAL_QCAR,
)
from pal.utilities.vision import Camera2D
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap

# --- V2X ADDITION ---
from pal.products.traffic_light import TrafficLight

# --- Networking Setup ---
COMPUTER_IP = "192.168.2.11"
PORT = 8080


# --- Camera Settings ---
CAMERA_ID = "3"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
FRAME_RATE = 30

# --- Controller Settings ---
tf = 6000
startDelay = 1
controllerUpdateRate = 100
v_ref = 0.5  # This is the cruise speed
K_p = 0.1
K_i = 1
enableSteeringControl = True
K_stanley = 1
nodeSequence = [10, 4, 20, 10]

# --- V2X ADDITION: Traffic Light & Geofencing Setup ---
TRAFFIC_LIGHT_IPS = ["192.168.2.15", "192.168.2.16"]
traffic_lights = [TrafficLight(ip) for ip in TRAFFIC_LIGHT_IPS]

# Geofencing areas
geofencing_threshold = 0.9
traffic_light_positions = [(2.367, 0.9246), (-2.1248, 1.0018)]


def generate_geofencing_areas(positions, threshold):
    return [
        {
            "name": f"Traffic Light {i+1}",
            "bounds": [(x - threshold, y - threshold), (x + threshold, y + threshold)],
        }
        for i, (x, y) in enumerate(positions)
    ]


def is_inside_geofence(position, geofence):
    (x_min, y_min), (x_max, y_max) = geofence
    return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max


geofencing_areas = generate_geofencing_areas(
    traffic_light_positions, geofencing_threshold
)
# --- End V2X ADDITION ---


# --- Global variables ---
is_running = True
latest_frame = None
frame_lock = threading.Lock()
# State variable from computer
car_state = "GO"
car_state_lock = threading.Lock()

# --- V2X ADDITION: Global state for traffic lights ---
traffic_light_statuses = ["UNKNOWN"] * len(TRAFFIC_LIGHT_IPS)
traffic_light_lock = threading.Lock()
has_stopped_at = {geofence["name"]: False for geofence in geofencing_areas}
# --- End V2X ADDITION ---


# --- Shutdown Signal Handler ---
def sig_handler(*args):
    global is_running
    is_running = False
    print("\nShutdown signal received.")


signal.signal(signal.SIGINT, sig_handler)


# --- Controller Classes (Unchanged) ---
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
        self.p_ref = (0, 0)
        self.th_ref = 0

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


# --- Thread Functions ---


# --- V2X ADDITION: Functions and Thread for V2X ---
def get_traffic_lights_status():
    """
    Fetch real-time status for multiple traffic lights.
    """
    try:
        status_map = {"1": "RED", "2": "YELLOW", "3": "GREEN"}
        statuses = []

        for i, light in enumerate(traffic_lights):
            status = light.status()
            status_str = status_map.get(status, "UNKNOWN")
            statuses.append(status_str)
        return statuses
    except Exception as e:
        print(f"Error fetching traffic light statuses: {e}")
        return ["UNKNOWN"] * len(traffic_lights)


def traffic_lights_status_thread_func():
    """Periodically polls traffic light statuses."""
    global is_running, traffic_light_statuses
    print("Traffic light status thread started...")
    while is_running:
        new_statuses = get_traffic_lights_status()
        with traffic_light_lock:
            traffic_light_statuses = new_statuses
        time.sleep(1)  # Poll once per second
    print("Traffic light status thread stopped.")


# --- End V2X ADDITION ---


# MODIFIED: Receiver thread now updates the shared state
def receiver_thread_func(sock):
    """Listens for commands from the computer and updates the shared car_state."""
    global is_running, car_state
    print("Receiver thread started...")
    while is_running:
        readable, _, _ = select.select([sock], [], [], 1.0)

        if sock in readable:
            try:
                data = sock.recv(1024)
                if data:
                    command = data.decode("utf-8").strip()
                    print(f"RECEIVED COMMAND: {command}")
                    if command in ["GO", "STOP"]:
                        with car_state_lock:
                            car_state = command
                else:
                    print("Receiver thread: Connection closed by computer.")
                    is_running = False
                    break
            except socket.error as e:
                if is_running:
                    print(f"Receiver thread: Socket error: {e}")
                break
    print("Receiver thread stopped.")


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


# --- V2X MODIFICATION: Control thread now reads V2X state AND computer state ---
def control_thread_func(initialPose, waypointSequence, calibrationPose, calibrate):
    """Runs the main vehicle control loop."""
    global is_running, car_state, traffic_light_statuses, has_stopped_at
    print("Control thread started...")

    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        gps = memoryview(b"")

    with qcar, gps:
        t0 = time.time()
        t = 0
        delta = 0
        while (t < tf + startDelay) and is_running:
            tp = t
            t = time.time() - t0
            dt = t - tp

            qcar.read()

            # --- V2X MODIFICATION: V2X Logic ---
            v2x_go_flag = True  # Assume we can go unless a light says no
            # --- End V2X MODIFICATION ---

            if enableSteeringControl:
                if gps.readGPS():
                    # --- V2X MODIFICATION: Check geofence and light status ---
                    position = (gps.position[0], gps.position[1])

                    # Read the global statuses safely
                    with traffic_light_lock:
                        current_statuses = traffic_light_statuses[
                            :
                        ]  # Make a local copy

                    for i, geofence in enumerate(geofencing_areas):
                        is_inside = is_inside_geofence(position, geofence["bounds"])

                        if is_inside:
                            # Check if we have a status for this light
                            if i < len(current_statuses):
                                traffic_light_status = current_statuses[i]

                                if (
                                    traffic_light_status == "RED"
                                    and not has_stopped_at[geofence["name"]]
                                ):
                                    v2x_go_flag = False  # STOP
                                    has_stopped_at[geofence["name"]] = True
                                    print(f"V2X: Stopping at {geofence['name']} (RED)")
                                    car_state = "STOP"

                                elif (
                                    traffic_light_status == "GREEN"
                                    and has_stopped_at[geofence["name"]]
                                ):
                                    v2x_go_flag = True  # RESUME
                                    has_stopped_at[geofence["name"]] = False
                                    print(
                                        f"V2X: Resuming at {geofence['name']} (GREEN)"
                                    )
                                    car_state = "GO"

                                # If already stopped and light is still red, we must remain stopped.

                        if not is_inside and has_stopped_at[geofence["name"]]:
                            has_stopped_at[geofence["name"]] = (
                                False  # Reset stop state when leaving area
                            )
                    # --- End V2X MODIFICATION ---

                    # Original EKF Update
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

                x, y, th = ekf.x_hat[0, 0], ekf.x_hat[1, 0], ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach

            if t < startDelay:
                u, delta = 0, 0
            else:
                # --- V2X MODIFICATION: Combined Speed Control Logic ---

                # 1. Get computer's desired state
                computer_go_flag = False
                with car_state_lock:
                    if car_state == "GO":
                        computer_go_flag = True

                # 2. V2X state is v2x_go_flag (calculated above)

                # 3. Combine: Must have "GO" from computer AND V2X
                target_speed = 0.0
                if computer_go_flag and v2x_go_flag:
                    target_speed = v_ref
                elif not computer_go_flag:
                    print("Control: Computer state is STOP", end="\r")
                elif not v2x_go_flag:
                    print(f"Control: V2X state is STOP", end="\r")

                u = speedController.update(v, target_speed, dt)
                # --- End V2X MODIFICATION ---

                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0

            qcar.write(u, delta)

        qcar.write(0, 0)
    is_running = False
    print("Control thread stopped.")


# --- Main Program ---
if not IS_PHYSICAL_QCAR:
    print("This script is designed to run on the physical QCar.")
else:
    if enableSteeringControl:
        roadmap = SDCSRoadMap(leftHandTraffic=False)
        waypointSequence = roadmap.generate_path(nodeSequence)
        initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
    else:
        initialPose = [0, 0, 0]

    # --- V2X ADDITION: Set traffic light timing ---

    # --- End V2X ADDITION ---

    calibrate = "y" in input("Do you want to recalibrate? (y/n): ")
    calibrationPose = [0, 2, -np.pi / 2]

    # --- Initialize Threads ---
    controlThread = None
    cameraThread = None
    receiverThread = None
    # --- V2X ADDITION ---
    trafficLightsThread = None
    # --- End V2X ADDITION ---
    client_socket = None
    camera = None

    try:
        # 1. Connect to the computer
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

        # 3. Start Threads
        cameraThread = threading.Thread(target=camera_thread_func, args=(camera,))
        controlThread = threading.Thread(
            target=control_thread_func,
            args=(initialPose, waypointSequence, calibrationPose, calibrate),
        )
        receiverThread = threading.Thread(
            target=receiver_thread_func, args=(client_socket,)
        )
        # --- V2X ADDITION ---
        trafficLightsThread = threading.Thread(target=traffic_lights_status_thread_func)
        # --- End V2X ADDITION ---

        cameraThread.start()
        controlThread.start()
        receiverThread.start()
        # --- V2X ADDITION ---
        trafficLightsThread.start()
        # --- End V2X ADDITION ---

        time.sleep(1.0)

        # 4. Main sending loop
        while is_running:
            local_frame = None
            with frame_lock:
                if latest_frame is not None:
                    local_frame = np.ascontiguousarray(latest_frame)

            if local_frame is not None:
                frame_bytes = local_frame.tobytes()
                message_header = struct.pack(">L", len(frame_bytes))
                client_socket.sendall(message_header)
                client_socket.sendall(frame_bytes)

            time.sleep(0.02)
    except Exception as e:
        print(f"An error occurred in the main thread: {e}")
    finally:
        print("Cleaning up resources...")
        is_running = False

        if controlThread and controlThread.is_alive():
            controlThread.join()
        if cameraThread and cameraThread.is_alive():
            cameraThread.join()
        if receiverThread and receiverThread.is_alive():
            receiverThread.join()
        # --- V2X ADDITION ---
        if trafficLightsThread and trafficLightsThread.is_alive():
            trafficLightsThread.join()
        # --- End V2X ADDITION ---

        if camera:
            camera.terminate()
        if client_socket:
            client_socket.close()

        print("Shutdown complete.")
