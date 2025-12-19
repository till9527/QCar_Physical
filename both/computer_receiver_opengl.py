# computer_receiver_cv2.py - Runs ON YOUR COMPUTER

import socket
import struct
import numpy as np
import time
import cv2
import threading
from pathlib import Path
from ultralytics import YOLO

# --- Settings ---
HOST = "0.0.0.0"
PORT = 8080
BASE_WIDTH = 640
BASE_HEIGHT = 480
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "model" / "best.pt"

# --- Globals ---
latest_frames = {}
frames_lock = threading.Lock()

# ### NEW: Dictionary to control client overrides from the main thread
# Structure: { addr: {'force_go': False} }
client_controls = {}
controls_lock = threading.Lock()


def receive_all(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            sock.settimeout(5.0)
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        except (socket.timeout, socket.error):
            return None
    return data


def handle_client(conn, addr, model):
    """
    Receives frames, runs inference, and obeys manual overrides.
    """
    print(f"Thread started for client: {addr}")

    # ### NEW: Register this client in the controls dictionary
    with controls_lock:
        client_controls[addr] = {"force_go": False}

    payload_size = struct.calcsize(">L")

    # --- State variables ---
    RED_LIGHT_COOLDOWN = 5
    last_start_time = 0
    car_state = "GO"

    # Stop Sign States
    is_stopped_for_sign = False
    stop_sign_start_time = 0
    yield_sign_start_time = 0
    is_stopped_for_yield = False
    STOP_SIGN_WAIT_TIME_S = 5.0
    STOP_SIGN_COOLDOWN_S = 10.0

    # Red Light / Obstacle States
    is_stopped_light = False
    last_stop_light_time = 0
    last_green_light_seen_time = 0
    found_pedestrian = False
    is_stopped_pedestrian = False
    last_pedestrian_seen_time = 0
    is_stopped_qcar = False
    last_qcar_seen_time = 0

    STOP_SIGN_MIN_WIDTH_THRESHOLD = 32

    try:
        while True:
            # --- 1. Receive Frame ---
            packed_msg_size = receive_all(conn, payload_size)
            if not packed_msg_size:
                break
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            frame_data = receive_all(conn, msg_size)
            if not frame_data:
                break

            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # --- 2. Inference ---
            results = model(frame, verbose=False, conf=0.3)
            annotated_frame = results[0].plot()

            current_time = time.time()

            # ### NEW: CHECK FOR MANUAL OVERRIDE ###
            force_go_active = False
            with controls_lock:
                if addr in client_controls:
                    force_go_active = client_controls[addr]["force_go"]

            if force_go_active:
                # If manual override is ON, force state to GO and reset stop timers
                if car_state == "STOP":
                    print(f"--- MANUAL OVERRIDE: Sending GO to {addr} ---")
                    conn.sendall(b"GO")
                    car_state = "GO"

                # Reset internal logic flags so it doesn't get stuck when we switch back to Auto
                is_stopped_pedestrian = False
                is_stopped_qcar = False
                is_stopped_light = False
                is_stopped_for_sign = False
                is_stopped_for_yield = False

                # Visual indicator on frame for the override
                cv2.putText(
                    annotated_frame,
                    "MANUAL OVERRIDE: GO",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Update shared frame and SKIP the rest of the logic
                with frames_lock:
                    latest_frames[addr] = annotated_frame
                continue

            # ----------------------------------------
            # ... If not overridden, proceed with normal Auto Logic ...
            # ----------------------------------------

            found_green_light = False
            found_red_light = False
            found_stop_sign = False
            found_yield_sign = False
            found_qcar = False
            found_pedestrian = False

            # (Parsing boxes - same as your original code)
            for box in results[0].boxes:
                class_name = model.names[int(box.cls[0].item())]
                xwyh = box.xywh[0]
                width = xwyh[2].item()
                height = xwyh[3].item()
                y = xwyh[1].item()
                x = xwyh[0].item()

                if class_name == "Qcar":
                    last_qcar_seen_time = current_time
                if class_name == "pedestrian":
                    last_pedestrian_seen_time = current_time
                if class_name == "Qcar" and height > 125:
                    found_qcar = True

                # Fixed pedestrian logic from your snippet
                if (
                    class_name == "pedestrian"
                    and (width > 50 or height > 100)
                    and 275 < x < 425
                ):
                    found_pedestrian = True

                if class_name == "red_light" and width > 15 and height < 50 and y < 200:
                    found_red_light = True

                if (
                    class_name == "green_light"
                    and width > 15
                    and height < 50
                    and y < 200
                ):
                    found_green_light = True
                    last_green_light_seen_time = current_time
                if class_name == "stop_sign" and width > STOP_SIGN_MIN_WIDTH_THRESHOLD:
                    found_stop_sign = True
                if class_name == "yield_sign" and width > STOP_SIGN_MIN_WIDTH_THRESHOLD:
                    found_yield_sign = True

            # --- Logic Block (Same as original) ---
            if found_pedestrian and car_state == "GO":
                print(f"--- {addr}: STOP (Pedestrian) ---")
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_pedestrian = True

            elif not found_pedestrian and car_state == "STOP" and is_stopped_pedestrian:
                print(f"--- {addr}: GO (Pedestrian Clear) ---")
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_pedestrian = False

            elif is_stopped_pedestrian and (
                current_time - last_pedestrian_seen_time > 5
            ):
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_pedestrian = False

            elif found_qcar and car_state == "GO":
                print(f"--- {addr}: STOP (QCar) ---")
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_qcar = True

            elif not found_qcar and car_state == "STOP" and is_stopped_qcar:
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_qcar = False

            elif is_stopped_qcar and (current_time - last_qcar_seen_time > 5):
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_qcar = False

            elif (
                found_red_light
                and car_state == "GO"
                and (current_time - last_start_time > RED_LIGHT_COOLDOWN)
                and (current_time - last_green_light_seen_time > 2)
            ):
                print(f"--- {addr}: STOP (Red Light) ---")
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_light = True
                last_stop_light_time = current_time

            elif (
                found_green_light
                and car_state == "STOP"
                and is_stopped_light
                and not is_stopped_for_sign
            ):
                print(f"--- {addr}: GO (Green Light) ---")
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_light = False
                last_start_time = current_time

            elif (
                found_stop_sign
                and car_state == "GO"
                and not is_stopped_for_sign
                and not is_stopped_light
                and (current_time - stop_sign_start_time > STOP_SIGN_COOLDOWN_S)
            ):
                print(f"--- {addr}: STOP (Stop Sign) ---")
                print("Width was: ", width)
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_for_sign = True
                stop_sign_start_time = current_time

            elif (
                found_yield_sign
                and car_state == "GO"
                and not is_stopped_for_sign
                and not is_stopped_light
                and (current_time - yield_sign_start_time > STOP_SIGN_COOLDOWN_S)
            ):
                print(f"--- {addr}: STOP (Yield Sign) ---")
                print("Width was: ", width)
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_for_yield = True
                yield_sign_start_time = current_time

            # Timeouts
            if (
                is_stopped_for_sign
                and car_state == "STOP"
                and (current_time - stop_sign_start_time > STOP_SIGN_WAIT_TIME_S)
                and not is_stopped_light
            ):
                print(f"--- {addr}: GO (Stop Wait Over) ---")
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_for_sign = False

            if (
                is_stopped_for_yield
                and car_state == "STOP"
                and (current_time - yield_sign_start_time > 3)
                and not is_stopped_light
            ):
                print(f"--- {addr}: GO (Yield Wait Over) ---")
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_for_yield = False

            with frames_lock:
                latest_frames[addr] = annotated_frame

    except Exception as e:
        print(f"Error for {addr}: {e}")
    finally:
        print(f"Closing {addr}.")
        with frames_lock:
            if addr in latest_frames:
                del latest_frames[addr]
        with controls_lock:
            if addr in client_controls:
                del client_controls[addr]
        conn.close()


def main():
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    def accept_connections(server_sock, yolo_model):
        while True:
            try:
                conn, addr = server_sock.accept()
                thread = threading.Thread(
                    target=handle_client, args=(conn, addr, yolo_model)
                )
                thread.daemon = True
                thread.start()
            except socket.error:
                break

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server on {PORT}.")
    print("--- CONTROLS ---")
    print("Press 'g' to FORCE GO (Manual Override)")
    print("Press 'a' to switch back to AUTO")
    print("Press 'q' to QUIT")
    print("----------------")

    accept_thread = threading.Thread(
        target=accept_connections, args=(server_socket, model)
    )
    accept_thread.daemon = True
    accept_thread.start()

    active_windows = set()
    running = True
    try:
        while running:
            with frames_lock:
                frames_to_show = latest_frames.copy()

            current_windows = set()
            for addr, frame in frames_to_show.items():
                window_name = f"QCar Feed {addr[0]}"
                cv2.imshow(window_name, frame)
                current_windows.add(window_name)
                active_windows.add(window_name)

            windows_to_close = active_windows - current_windows
            for window_name in windows_to_close:
                cv2.destroyWindow(window_name)
            active_windows = current_windows

            # ### NEW: Keyboard Input Handling ###
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                running = False

            elif key == ord("g"):
                print(">>> UI COMMAND: Enabling Force GO")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_go"] = True

            elif key == ord("a"):
                print(">>> UI COMMAND: Re-enabling AUTO Mode")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_go"] = False

            if not accept_thread.is_alive() and not active_windows:
                # Keep running if thread is alive but no clients yet
                if not accept_thread.is_alive():
                    running = False

    finally:
        print("\nShutting down.")
        server_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
