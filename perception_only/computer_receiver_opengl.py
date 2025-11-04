# computer_receiver_cv2.py - Runs ON YOUR COMPUTER

import socket
import struct
import numpy as np
import time
import cv2  # MODIFIED: Using OpenCV for display
import threading

from ultralytics import YOLO

# --- Settings ---
HOST = "0.0.0.0"
PORT = 8080
BASE_WIDTH = 320
BASE_HEIGHT = 240

# --- NEW: Global, thread-safe dictionary to hold the latest frame from each client ---
# This allows the main thread to access frames from all worker threads.
latest_frames = {}
frames_lock = threading.Lock()

from pal.products.traffic_light import TrafficLight


TRAFFIC_LIGHT_IPS = ["192.168.2.19", "192.168.2.18"]
traffic_lights = [TrafficLight(ip) for ip in TRAFFIC_LIGHT_IPS]

for traffic_light in traffic_lights:
    traffic_light.timed(red=20, yellow=5, green=10)


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
def handle_client(conn, addr, model):
    """
    Receives frames from a QCar, runs inference, sends commands,
    and places the annotated frame in a shared dictionary for display.
    """
    print(f"Thread started for client: {addr}")
    payload_size = struct.calcsize(">L")

    # State variables are local to this thread
    RED_LIGHT_COOLDOWN = 5
    last_start_time = 0
    car_state = "GO"

    try:
        while True:
            # Receive frame
            packed_msg_size = receive_all(conn, payload_size)
            if not packed_msg_size:
                break
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            frame_data = receive_all(conn, msg_size)
            if not frame_data:
                break
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (BASE_HEIGHT, BASE_WIDTH, 3)
            )

            # Run YOLO and process logic
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            # Command logic (unchanged)
            found_green_light = False
            found_red_light_condition = False
            for box in results[0].boxes:
                class_name = model.names[int(box.cls[0].item())]
                xwyh = box.xywh[0]
                width = xwyh[2].item()
                # print(width)
                if (
                    class_name == "red_light"
                    and (time.time() - last_start_time > RED_LIGHT_COOLDOWN)
                    and width > 8
                ):
                    found_red_light_condition = True

                if class_name == "green_light":
                    found_green_light = True

            if found_red_light_condition and car_state == "GO":
                print(f"--- COMMAND to {addr}: Sending STOP ---")
                conn.sendall(b"STOP")
                car_state = "STOP"

            elif found_green_light and car_state == "STOP":
                print(f"--- COMMAND to {addr}: Sending GO ---")
                conn.sendall(b"GO")
                car_state = "GO"
                last_start_time = time.time()

            # Put the processed frame into the shared dictionary
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


# --- MODIFIED: Main application logic using OpenCV ---
def main():
    print("Loading YOLOv8 model...")
    model_path = "../model/best.pt"
    model = YOLO(model_path)
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
