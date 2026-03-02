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

# --- NEW: Global, thread-safe dictionary to hold the latest frame from each client ---
# This allows the main thread to access frames from all worker threads.
latest_frames = {}
frames_lock = threading.Lock()

# Manual override controls per client
# Structure: { addr: {'force_go': False, 'force_stop': False} }
client_controls = {}
controls_lock = threading.Lock()


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


def handle_client(conn, addr):
    """
    Receives frames from a QCar, runs inference, sends commands,
    and places the annotated frame in a shared dictionary for display.
    """
    print(f"Thread started for client: {addr}")

    with controls_lock:
        client_controls[addr] = {"force_go": False, "force_stop": False}

    payload_size = struct.calcsize(">L")

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

            force_go_active = False
            force_stop_active = False
            with controls_lock:
                if addr in client_controls:
                    force_go_active = client_controls[addr]["force_go"]
                    force_stop_active = client_controls[addr]["force_stop"]

            if force_stop_active:
                try:
                    conn.sendall(b"FORCE_STOP")
                except socket.error:
                    break
                cv2.putText(
                    frame,
                    "MANUAL OVERRIDE: STOP",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            elif force_go_active:
                try:
                    conn.sendall(b"FORCE_GO")
                except socket.error:
                    break
                cv2.putText(
                    frame,
                    "MANUAL OVERRIDE: GO",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # --- 5. Update Shared Frame ---
            with frames_lock:
                latest_frames[addr] = frame

    except Exception as e:
        print(f"An error occurred in thread for {addr}: {e}")
    finally:
        print(f"Closing connection for {addr}.")
        # Clean up the dictionary when a client disconnects
        with frames_lock:
            if addr in latest_frames:
                del latest_frames[addr]
        with controls_lock:
            if addr in client_controls:
                del client_controls[addr]
        conn.close()


# ... (keep the rest of the file, including main(), as-is) ...
# --- MODIFIED: Main application logic using OpenCV ---
def main():

    # --- Listener thread setup (Unchanged) ---
    def accept_connections(server_sock):
        while True:
            try:
                conn, addr = server_sock.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
            except socket.error:
                break  # Exit if server_socket is closed

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server listening on port {PORT}.")
    print("--- CONTROLS ---")
    print("Press 'g' to FORCE GO (Manual Override)")
    print("Press 's' to FORCE STOP (Manual Override)")
    print("Press 'a' to switch back to AUTO")
    print("Press 'q' to QUIT")
    print("----------------")

    accept_thread = threading.Thread(target=accept_connections, args=(server_socket,))
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

            # Wait for keyboard input.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                running = False

            elif key == ord("g"):
                print(">>> UI COMMAND: Enabling Force GO")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_go"] = True
                        client_controls[addr]["force_stop"] = False

            elif key == ord("s"):
                print(">>> UI COMMAND: Enabling Force STOP")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_stop"] = True
                        client_controls[addr]["force_go"] = False

            elif key == ord("a"):
                print(">>> UI COMMAND: Re-enabling AUTO Mode")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_go"] = False
                        client_controls[addr]["force_stop"] = False

            # If the server thread stops, we should also exit
            if not accept_thread.is_alive() and not active_windows:
                running = False

    finally:
        print("\nShutting down server.")
        server_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
