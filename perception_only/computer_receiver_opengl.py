# computer_receiver_cv2.py - Runs ON YOUR COMPUTER

import socket
import struct
import numpy as np
import time
import cv2  # MODIFIED: Using OpenCV for display
import threading
from pathlib import Path
from ultralytics import YOLO

# --- Settings ---
HOST = "0.0.0.0"
PORT = 8080
BASE_WIDTH = 640
BASE_HEIGHT = 480
SCRIPT_DIR = Path(__file__).resolve().parent
# This joins that directory path with your model path
MODEL_PATH = SCRIPT_DIR.parent / "model" / "best.pt"
# --- NEW: Global, thread-safe dictionary to hold the latest frame from each client ---
# This allows the main thread to access frames from all worker threads.
latest_frames = {}
frames_lock = threading.Lock()


# --- Helper function (Unchanged) ---
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


# --- Worker thread for handling a single client ---
# computer_receiver_opengl.py

# ... (keep all your imports and other functions) ...


# --- Worker thread for handling a single client ---
def handle_client(conn, addr, model):
    """
    Receives frames from a QCar, runs inference, sends commands,
    and places the annotated frame in a shared dictionary for display.
    """
    print(f"Thread started for client: {addr}")
    payload_size = struct.calcsize(">L")

    # --- State variables ---
    RED_LIGHT_COOLDOWN = 5
    last_start_time = 0  # For red lights
    car_state = "GO"  # The *command* we last sent

    # NEW: Cleaned-up state variables for stop signs
    is_stopped_for_sign = False  # Is the car *currently* stopped for a sign?
    stop_sign_start_time = 0  # Time the last STOP command was sent
    yield_sign_start_time = 0
    is_stopped_for_yield = False
    STOP_SIGN_WAIT_TIME_S = 5.0
    STOP_SIGN_COOLDOWN_S = 10.0  # Time to ignore signs *after stopping*

    # --- Red light state ---
    is_stopped_light = False
    last_stop_light_time = 0
    last_green_light_seen_time = 0

    # NEW: Use the width threshold from your original file
    STOP_SIGN_MIN_WIDTH_THRESHOLD = 32
    RED_LIGHT_WAIT_TIME = 20

    try:
        while True:
            # --- 1. Receive Frame and Run Inference ---
            packed_msg_size = receive_all(conn, payload_size)
            if not packed_msg_size:
                break
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            frame_data = receive_all(conn, msg_size)
            if not frame_data:
                break

            # CHANGED: Decode the JPEG byte array back to an image
            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Safety check in case decoding fails
            if frame is None:
                continue

            results = model(frame, verbose=False, conf=0.65)
            annotated_frame = results[0].plot()

            # --- 2. Process Detections ---
            # We must loop through all detections to find all objects
            current_time = time.time()
            found_green_light = False
            found_red_light = False
            found_stop_sign = False
            found_yield_sign = False

            for box in results[0].boxes:
                class_name = model.names[int(box.cls[0].item())]
                xwyh = box.xywh[0]
                width = xwyh[2].item()
                height = xwyh[3].item()
                y = xwyh[1].item()
                # if class_name == "red light":
                #     # print("Width: ", width)
                #     # print("Height: ", height)

                if class_name == "red_light" and width > 13 and height < 50 and y < 200:
                    found_red_light = True

                if (
                    class_name == "yellow_light"
                    and width > 13
                    and height < 50
                    and y < 200
                ):
                    found_red_light = True

                if (
                    class_name == "green_light"
                    and width > 13
                    and height < 50
                    and y < 200
                ):
                    found_green_light = True
                    last_green_light_seen_time = current_time

                if class_name == "stop_sign" and width > STOP_SIGN_MIN_WIDTH_THRESHOLD:
                    found_stop_sign = True
                if class_name == "yield_sign" and width > STOP_SIGN_MIN_WIDTH_THRESHOLD:
                    found_yield_sign = True
            # --- 3. Detection-Based Action Logic ---
            # This block handles actions based on *seeing* an object
            # NEW: This logic is now a clean if/elif chain for priority

            # A. Red Light Stop
            if (
                found_red_light
                and car_state == "GO"  # NEW: Only stop if we're not already stopped
                and (current_time - last_start_time > RED_LIGHT_COOLDOWN)
                and (current_time - last_green_light_seen_time > 2)
            ):
                print(f"--- COMMAND to {addr}: Sending STOP (Red Light) ---")
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_light = True
                last_stop_light_time = current_time

            # B. Green Light Go
            elif (
                found_green_light
                and car_state == "STOP"  # NEW: Only go if we're currently stopped
                and is_stopped_light  # Only go if we were stopped for a light
                and not is_stopped_for_sign  # But not if we're in a stop sign wait
            ):
                print(f"--- COMMAND to {addr}: Sending GO (Green Light) ---")
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_light = False
                last_start_time = current_time  # Reset red light cooldown

            # C. Stop Sign Stop
            elif (
                found_stop_sign
                and car_state == "GO"  # NEW: Only stop if we are moving
                and not is_stopped_for_sign  # And not already stopped for a sign
                and not is_stopped_light  # Red light has priority
                and (
                    current_time - stop_sign_start_time > STOP_SIGN_COOLDOWN_S
                )  # And 10s cooldown has passed
            ):
                print(f"--- COMMAND to {addr}: Sending STOP (Stop Sign) ---")
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_for_sign = True  # We are now in a stop-sign-wait
                stop_sign_start_time = current_time  # Record the time we stopped
            elif (
                found_yield_sign
                and car_state == "GO"  # NEW: Only stop if we are moving
                and not is_stopped_for_sign  # And not already stopped for a sign
                and not is_stopped_light  # Red light has priority
                and (
                    current_time - yield_sign_start_time > STOP_SIGN_COOLDOWN_S
                )  # And 10s cooldown has passed
            ):
                print(f"--- COMMAND to {addr}: Sending STOP (Yield Sign) ---")
                conn.sendall(b"STOP")
                car_state = "STOP"
                is_stopped_for_yield = True  # We are now in a stop-sign-wait
                yield_sign_start_time = current_time  # Record the time we stopped

            # --- 4. Timeout-Based Action Logic ---
            # NEW: This block is separate and handles the 5-second wait

            # A. Stop Sign Resume (after 5 seconds)
            if (
                is_stopped_for_sign  # We are stopped for a sign
                and car_state == "STOP"  # NEW: We are currently stopped
                and (
                    current_time - stop_sign_start_time > STOP_SIGN_WAIT_TIME_S
                )  # And 5 seconds have passed
                and not is_stopped_light  # And no red light is present
            ):
                print(f"--- COMMAND to {addr}: Sending GO (Stop Sign Wait Over) ---")
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_for_sign = False

            if (
                is_stopped_for_yield  # We are stopped for a sign
                and car_state == "STOP"  # NEW: We are currently stopped
                and (
                    current_time - yield_sign_start_time > 3
                )  # And 5 seconds have passed
                and not is_stopped_light  # And no red light is present
            ):
                print(f"--- COMMAND to {addr}: Sending GO (Yield Sign Wait Over) ---")
                conn.sendall(b"GO")
                car_state = "GO"
                is_stopped_for_yield = False

            # --- 5. Update Shared Frame ---
            with frames_lock:
                latest_frames[addr] = annotated_frame

    except Exception as e:
        print(f"An error occurred in thread for {addr}: {e}")
    finally:
        print(f"Closing connection for {addr}.")
        # Clean up the dictionary when a client disconnects
        with frames_lock:
            if addr in latest_frames:
                del latest_frames[addr]
        conn.close()


# ... (keep the rest of the file, including main(), as-is) ...
# --- MODIFIED: Main application logic using OpenCV ---
def main():
    print("Loading YOLOv8 model...")

    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    # --- Listener thread setup (Unchanged) ---
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
                break  # Exit if server_socket is closed

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server listening on port {PORT}... Press 'q' in any video window to quit.")

    accept_thread = threading.Thread(
        target=accept_connections, args=(server_socket, model)
    )
    accept_thread.daemon = True
    accept_thread.start()

    # --- NEW: Main Thread GUI Management Loop using OpenCV ---
    active_windows = set()
    running = True
    try:
        while running:
            # Create a copy of the dictionary to avoid issues if it changes during iteration
            with frames_lock:
                frames_to_show = latest_frames.copy()

            # Display a window for each connected client
            current_windows = set()
            for addr, frame in frames_to_show.items():
                window_name = f"QCar Feed from {addr[0]}"
                cv2.imshow(window_name, frame)
                current_windows.add(window_name)
                active_windows.add(window_name)

            # Check for and close any windows for clients that have disconnected
            windows_to_close = active_windows - current_windows
            for window_name in windows_to_close:
                cv2.destroyWindow(window_name)
            active_windows = current_windows

            # Wait for a key press. This also allows OpenCV to process its GUI events.
            # Press 'q' to quit the entire application.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

            # If the server thread stops, we should also exit
            if not accept_thread.is_alive() and not active_windows:
                running = False

    finally:
        print("\nShutting down server.")
        server_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
