# --------------------------------------------------------------------------------------
# Project: OpenMicroManipulator
# License: MIT (see LICENSE file for full description)
#          All text in here must be included in any redistribution.
# Author:  M. S. (diffraction limited)
# --------------------------------------------------------------------------------------

import glob
import re
import sys
import threading
import time

import cv2
import numpy as np

from hardware.abstract_camera import AbstractCamera


class OpenCVCamera(AbstractCamera):
    @staticmethod
    def enumeration_indices(max_indices):
        if sys.platform.startswith("linux"):
            indices = sorted(
                int(match.group(1))
                for path in glob.glob("/dev/video*")
                if (match := re.search(r"(\d+)$", path))
            )
            if indices:
                return indices

        return list(range(max_indices))

    @staticmethod
    def _enumeration_backends():
        if sys.platform.startswith("win"):
            # Only DirectShow: probing missing indices with MSMF / Auto takes
            # seconds per index and made startup very slow.
            return [
                (cv2.CAP_DSHOW, "DirectShow"),
            ]

        if sys.platform == "darwin":
            return [
                (cv2.CAP_AVFOUNDATION, "AVFoundation"),
                (cv2.CAP_ANY, "Auto"),
            ]

        return [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Auto"),
        ]

    @staticmethod
    def enumerate_devices(max_indices=8):
        print("Enumerating OpenCV cameras (driver errors below are expected and can be ignored)...")
        devices = []

        # Without a device list (Windows / macOS) indices are probed blindly. Cameras
        # use consecutive indices, so stop at the first index without a camera.
        stop_at_first_gap = not sys.platform.startswith("linux")

        previous_log_level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        try:
            for index in OpenCVCamera.enumeration_indices(max_indices):
                device_count = len(devices)
                for backend, backend_label in OpenCVCamera._enumeration_backends():
                    capture = None
                    try:
                        capture = cv2.VideoCapture(index, backend)
                        if not capture.isOpened():
                            continue

                        ok, _ = capture.read()
                        if not ok:
                            continue

                        devices.append({
                            "kind": "opencv",
                            "id": f"opencv:{index}:{backend}",
                            "label": f"OpenCV Camera {index}",
                            "index": index,
                            "backend": backend,
                            "backend_label": backend_label,
                        })
                        break
                    except Exception:
                        continue
                    finally:
                        if capture is not None:
                            capture.release()

                if stop_at_first_gap and len(devices) == device_count:
                    break
        finally:
            cv2.utils.logging.setLogLevel(previous_log_level)

        return devices

    def __init__(self, camera_index=0, backend=cv2.CAP_ANY, resolution=None):
        self.camera_index = int(camera_index)
        self.backend = backend
        self.cap = cv2.VideoCapture(self.camera_index, self.backend)
        self.grabbing = False
        self.dark_image = None
        # If no frame can be grabbed for this long, treat the camera as disconnected.
        self.grab_failure_timeout_s = 2.0
        # VideoCapture is not thread safe: the grab loop runs in a worker thread while
        # settings are changed from the GUI thread, so all cap access goes through this lock.
        self._cap_lock = threading.Lock()

        if not self.cap.isOpened():
            print(f"Failed to open OpenCV camera at index {self.camera_index}")
            self.cap.release()
            self.cap = None
            return

        print(f"Using OpenCV camera at index {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if resolution is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4100)

    @staticmethod
    def _convert_frame(frame):
        if frame is None:
            return None

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def is_connected(self):
        return self.cap is not None

    def set_gain(self, gain: float):
        with self._cap_lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_GAIN, gain)

    def set_white_balance(self, wb: float):
        with self._cap_lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, int(wb))

    def get_exposure_time_range(self):
        return 0, 1

    def set_exposure_time(self, exposure_time_us):
        with self._cap_lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_time_us))

    def capture_dark_image(self, n_frames=10):
        if not self.cap:
            return
        with self._cap_lock:
            prev_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        self.set_exposure_time(0)
        frames = []
        for _ in range(n_frames):
            with self._cap_lock:
                ret, frame = self.cap.read() if self.cap else (False, None)
            if ret:
                frames.append(self._convert_frame(frame).astype(np.float32))
        self.dark_image = np.mean(frames, axis=0) if frames else None
        self.set_exposure_time(prev_exposure)

    def start_grabbing(self, single_grab=True):
        self.grabbing = True

    def stop_grabbing(self):
        self.grabbing = False

    def grab_single_triggered(self, timeout_ms=1000):
        return self.grab_one(timeout_ms)

    def grab_one(self, timeout_ms=5000):
        with self._cap_lock:
            if not self.cap:
                return None
            ret, frame = self.cap.read()
        if not ret:
            return None
        frame = self._convert_frame(frame)
        if self.dark_image is not None and frame is not None:
            frame = np.clip(frame.astype(np.float32) - self.dark_image, 0, 255).astype(np.uint8)
        return frame

    def grab_loop(self, callback, timeout_ms=5000):
        if not self.cap:
            return

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 60
        self.start_grabbing(single_grab=False)
        failure_deadline = None
        try:
            while self.grabbing:
                new_frame = self.grab_one(timeout_ms)
                if new_frame is not None:
                    failure_deadline = None
                    # rotate by 180 deg (same as flipping both axes) in one fast pass
                    frame = cv2.flip(new_frame, -1)
                else:
                    # No frame received. The camera may have been unplugged, in which
                    # case reads keep failing forever. Allow a short grace period, then
                    # treat the camera as disconnected so the stream can shut down
                    # cleanly instead of spinning (or blocking) indefinitely.
                    now = time.monotonic()
                    if failure_deadline is None:
                        failure_deadline = now + self.grab_failure_timeout_s
                    elif now >= failure_deadline:
                        raise RuntimeError("OpenCV camera disconnected (no frames received).")
                    # Avoid busy-spinning while the camera is unavailable.
                    time.sleep(0.05)

                try:
                    if callback(frame) is False:
                        break
                except Exception as exc:
                    print(f"Error in callback: {exc}")
        finally:
            self.stop_grabbing()

    def close(self):
        self.stop_grabbing()
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

    def __del__(self):
        self.close()
