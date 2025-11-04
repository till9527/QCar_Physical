# qcar_local_control_and_record.py - Runs ONLY ON THE QCAR
# No networking. Controls the car and records video locally.

import threading
import time
import signal
import numpy as np
import cv2  # For video writing

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
from pal.products.traffic_light import TrafficLight

TRAFFIC_LIGHT_IPS = ["192.168.2.12", "192.168.2.11"]
traffic_lights = [TrafficLight(ip) for ip in TRAFFIC_LIGHT_IPS]

# Set a timed sequence for the physical traffic lights
for traffic_light in traffic_lights:
    traffic_light.timed(red=20, yellow=5, green=10)

# --- Camera Settings ---
CAMERA_ID = "3"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30

# --- Controller Settings ---
tf = 6000
startDelay = 1
controllerUpdateRate = 100
v_ref = 0.5  # Car will now just use this reference speed
K_p = 0.1
K_i = 1
enableSteeringControl = True
K_stanley = 1
nodeSequence = [10, 4, 20, 10]

# --- Global variables ---
is_running = True
# --- REMOVED: All network-related globals (latest_frame, car_state, etc.) ---


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

# --- REMOVED: receiver_thread_func ---


# --- MODIFIED: Camera thread now ONLY writes to a video file ---
def camera_thread_func(camera, video_writer):
    """
    Continuously reads from the camera and writes to a local video file.
    """
    global is_running
    print("Camera thread started...")
    while is_running:
        if camera.read():
            # Write the frame to the local video file
            video_writer.write(camera.imageData)

        # We sleep here to match the camera's frame rate
        time.sleep(1 / FRAME_RATE)
    print("Camera thread stopped.")


# --- MODIFIED: Control thread now runs independently ---
def control_thread_func(initialPose, waypointSequence, calibrationPose, calibrate):
    """Runs the main vehicle control loop."""
    global is_running
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
            if enableSteeringControl:
                if gps.readGPS():
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
                # --- MODIFIED: Removed state-based logic. Car just uses v_ref ---
                u = speedController.update(v, v_ref, dt)

                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0

            qcar.write(u, delta)

        qcar.write(0, 0)

    is_running = False  # Stop other threads when control loop finishes
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

    calibrate = "y" in input("Do you want to recalibrate? (y/n): ")
    calibrationPose = [0, 2, -np.pi / 2]

    # --- Initialize Threads ---
    controlThread = None
    cameraThread = None
    # --- REMOVED: receiverThread and client_socket ---
    camera = None

    # --- Initialize VideoWriter variables ---
    out = None
    video_filename = ""

    try:
        # --- REMOVED: All socket connection logic ---

        # 1. Initialize Camera
        camera = Camera2D(
            cameraId=CAMERA_ID,
            frameWidth=IMAGE_WIDTH,
            frameHeight=IMAGE_HEIGHT,
            frameRate=FRAME_RATE,
        )

        # 2. Set up the local VideoWriter
        timestr = time.strftime("%Y%m%d-%H%M%S")
        video_filename = f"qcar_local_record_{timestr}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            video_filename, fourcc, FRAME_RATE, (IMAGE_WIDTH, IMAGE_HEIGHT)
        )
        print(f"Recording video locally to {video_filename}")

        # 3. Start Threads
        cameraThread = threading.Thread(target=camera_thread_func, args=(camera, out))
        controlThread = threading.Thread(
            target=control_thread_func,
            args=(initialPose, waypointSequence, calibrationPose, calibrate),
        )
        # --- REMOVED: receiverThread ---

        cameraThread.start()
        controlThread.start()

        # 4. Main loop
        # The main thread just needs to stay alive to catch the Ctrl+C signal
        print("Control and recording loops are running. Press Ctrl+C to stop.")
        while is_running:
            time.sleep(1.0)  # Keep main thread alive

    except Exception as e:
        print(f"An error occurred in the main thread: {e}")
    finally:
        print("Cleaning up resources...")
        is_running = False  # Ensure all threads know to stop

        if controlThread and controlThread.is_alive():
            controlThread.join()
        if cameraThread and cameraThread.is_alive():
            cameraThread.join()
        # --- REMOVED: receiverThread.join() ---

        # Release the VideoWriter to save the file
        if out:
            out.release()
            print(f"Local video recording saved: {video_filename}")

        if camera:
            camera.terminate()
        # --- REMOVED: client_socket.close() ---

        print("Shutdown complete.")
