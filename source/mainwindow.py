# --------------------------------------------------------------------------------------
# Project: OpenMicroManipulator
# License: MIT (see LICENSE file for full description)
#          All text in here must be included in any redistribution.
# Author:  M. S. (diffraction limited)
# --------------------------------------------------------------------------------------

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from gcode_runner import GCodeRunner
from gui_components.image_viewer_widget import ImageViewerWidget
from gui_components.realtime_controller_widget import RealtimeControllerWidget
from hardware.camera_basler import BaslerCamera
from hardware.camera_opencv import OpenCVCamera
from hardware.device_discovery import list_camera_devices, list_serial_devices
from hardware.open_micro_stage_api import OpenMicroStageInterface, SerialInterface
from image_processing.image_point_tracker import ImagePointTracker
from ui_loader import load_ui_type
from version import __version__
from PySide6.QtCore import Qt, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

_ui_path = os.path.join(os.path.dirname(__file__), "mainwindow.ui")
Ui_DeviceControlMainWindow = load_ui_type(_ui_path)


class CameraStreamWorker(QThread):
    frame_ready = Signal(object)
    stream_error = Signal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.running = True
        # True while the GUI thread has not yet processed the last emitted frame.
        # New frames are dropped meanwhile so they cannot pile up in the event queue.
        self.frame_pending = False

    def frame_consumed(self):
        self.frame_pending = False

    def stop(self):
        self.running = False
        try:
            self.camera.stop_grabbing()
        except Exception:
            pass
        self.wait(1500)

    def _handle_frame(self, frame):
        if not self.running:
            return False

        if self.frame_pending:
            return True  # GUI still busy with the previous frame, drop this one

        self.frame_pending = True
        self.frame_ready.emit(frame)
        return True

    def run(self):
        try:
            self.camera.grab_loop(callback=self._handle_frame, timeout_ms=500)
        except Exception as exc:
            if self.running:
                self.stream_error.emit(str(exc))


class BackgroundTask(QThread):
    """Runs a blocking function (e.g. a slow serial command) outside the GUI thread."""
    task_done = Signal(object, object)  # (result, exception or None)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            result = self.function()
        except Exception as exc:
            self.task_done.emit(None, exc)
            return
        self.task_done.emit(result, None)


class PositionPollWorker(QThread):
    """
    Periodically reads target and measured position (firmware command M54) in a
    background thread. While another command is in progress (homing, G-code, ...) the
    read simply waits for its turn, so the GUI thread is never blocked.
    """
    position_updated = Signal(object, object)  # (target, measured) in device coordinates, or None
    unsupported = Signal()                     # firmware does not answer M54

    def __init__(self, oms, interval_s=0.2, parent=None):
        super().__init__(parent)
        self.oms = oms
        self.interval_s = interval_s
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        # Probe with a short timeout: firmware without M54 does not reply at all.
        res, target, measured = self.oms.read_target_and_measured_position(timeout_s=2)
        if not self.running:
            return
        if res != SerialInterface.ReplyStatus.OK:
            self.unsupported.emit()
            return
        self.position_updated.emit(target, measured)

        while self.running and self.oms.is_connected():
            deadline = time.monotonic() + self.interval_s
            while self.running and time.monotonic() < deadline:
                time.sleep(0.02)
            if not self.running:
                break

            # long timeout: the device answers late while a queued move blocks its input
            res, target, measured = self.oms.read_target_and_measured_position(timeout_s=60)
            if self.running and res == SerialInterface.ReplyStatus.OK:
                self.position_updated.emit(target, measured)


class CalibrationPlotDialog(QDialog):
    """Shows the raw encoder counts over motor angle for each calibrated joint.

    Mirrors the visualisation done by the standalone calibration_plotter.py, but
    embeds the matplotlib canvas in a Qt dialog instead of opening a blocking window.
    """

    def __init__(self, joint_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axis Calibration")
        self.resize(900, 640)

        # Imported lazily so the application still starts if matplotlib is missing.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.figure import Figure

        canvas = FigureCanvasQTAgg(Figure(figsize=(10, 7)))
        ax = canvas.figure.subplots(1, 1)

        for joint_index, data in joint_data:
            # data[0]: motor angles, data[2]: raw encoder counts
            ax.plot(data[0], data[2], label=f"Actuator {joint_index}")

        ax.set_xlabel("Motor Angle [rad]")
        ax.set_ylabel("Encoder Counts Raw")
        ax.set_title("Encoder Count Plot")
        ax.legend()
        ax.grid(True)
        canvas.figure.tight_layout()

        saved_label = QLabel("Calibration saved to device")
        saved_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(canvas, self))
        layout.addWidget(canvas)
        layout.addWidget(saved_label)


class DeviceControlMainWindow(QMainWindow, Ui_DeviceControlMainWindow):
    # emitted from the G-code runner thread, handled in the GUI thread
    gcode_finished_signal = Signal()

    def __init__(self, oms: OpenMicroStageInterface, camera=None):
        super().__init__()

        self.oms = oms
        self.camera = camera
        self.camera_stream_worker = None
        self.gcode_runner = None
        self.stage_task = None  # BackgroundTask while homing / calibrating / sending a path
        self.position_poll_worker = None

        self.settings = QSettings()
        self.pixel_per_mm = 2000.0/2.2
        self.connected_stage_label = None
        self.connected_camera_label = None
        self.serial_devices = []
        self.camera_devices = []
        self.stage_dependent_widgets = []
        self.camera_dependent_widgets = []

        self.last_frame = None
        self.draw_buffer = None
        self.current_pos = [0, 0, 0]
        self.step_sizes = [1.0, 0.1, 0.01, 0.001, 0.0001]
        self.feedrates = [50.0, 5.0, 5.0, 1.0, 0.1]
        self.step_size_idx = 1
        self.waypoints = []

        self.image_point_tracker = ImagePointTracker()

        self.init_ui()
        self.refresh_serial_devices()
        self.refresh_camera_devices()
        self.show_placeholder_frame()
        self.update_connection_state()

        if self.camera is not None and self.camera.is_connected():
            self.connected_camera_label = "Configured Camera"
            self.load_camera_settings(self.connected_camera_label)
            self.apply_camera_settings()
            self.start_camera_stream()

        if self.oms.is_connected():
            self.connected_stage_label = "Configured Device"
            self.on_stage_connected()

    def init_ui(self):
        self.setupUi(self)
        self.setWindowTitle(f"{self.windowTitle()}  v{__version__}")

        self.video_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.create_position_display()

        # Connect signals – connection buttons
        self.serial_refresh_button.clicked.connect(self.refresh_serial_devices)
        self.serial_connect_button.clicked.connect(self.toggle_stage_connection)
        self.camera_refresh_button.clicked.connect(self.refresh_camera_devices)
        self.camera_connect_button.clicked.connect(self.toggle_camera_connection)

        # Connect signals – movement buttons
        for name, axis, direction in [
            ("btn_y_minus", 1, -1), ("btn_z_plus", 2, +1),
            ("btn_x_minus", 0, -1), ("btn_x_plus", 0, +1),
            ("btn_y_plus", 1, +1), ("btn_z_minus", 2, -1),
        ]:
            btn = getattr(self, name)
            btn.clicked.connect(lambda checked=False, a=axis, d=direction: self.move_axis(a, d))

        # Spinbox signals
        self.accel_spinbox.editingFinished.connect(self.apply_acceleration_setting)

        # (tool index, value spinbox, on/off button) for every tool output.
        self.tool_controls = [
            (0, self.tool1_spinbox, self.btn_tool1_on),
            (1, self.tool2_spinbox, self.btn_tool2_on),
        ]
        for tool_idx, spinbox, button in self.tool_controls:
            button.toggled.connect(lambda _checked=False, i=tool_idx: self.apply_tool_setting(i))
            spinbox.valueChanged.connect(lambda _value=0.0, i=tool_idx: self.apply_tool_setting(i))

        # Camera settings are applied shortly after the last change instead of on every
        # single step, because each camera call can block for a moment.
        self.camera_settings_timer = QTimer(self)
        self.camera_settings_timer.setSingleShot(True)
        self.camera_settings_timer.setInterval(250)
        self.camera_settings_timer.timeout.connect(self.apply_camera_settings)
        # (lambda so the spinbox value is not passed on as the timer interval)
        for spinbox in (self.exposure_spinbox, self.gain_spinbox, self.white_balance_spinbox):
            spinbox.valueChanged.connect(lambda _value=0.0: self.camera_settings_timer.start())

        self.gcode_finished_signal.connect(self.on_gcode_runner_finished)

        # Step size buttons
        self.step_button_group = QButtonGroup()
        self.step_button_group.setExclusive(True)
        for i in range(len(self.step_sizes)):
            btn = getattr(self, f"step_btn_{i}")
            btn.clicked.connect(lambda checked=False, idx=i: self.set_step_size(idx))
            self.step_button_group.addButton(btn, i)

        self.realtime_control_widget.setup(self.video_viewer.viewport(), self.oms)
        self.realtime_control_widget.stop_control_signal.connect(self.on_stop_realtime_control)

        self.register_stage_widget(self.device_controls_frame)
        self.register_stage_widget(self.AdvancedTab)
        self.register_stage_widget(self.PathTab)
        self.register_camera_widget(self.CameraTab)

        # Advanced tab
        self.btn_3point_alignment.clicked.connect(self.run_3point_alignment)
        self.btn_set_origin.clicked.connect(self.set_origin)
        self.btn_set_tracking.clicked.connect(self.set_tracking_point)
        self.btn_clear.clicked.connect(self.clear_draw_buffer)
        self.btn_load_transform.clicked.connect(self.load_transform)
        self.btn_save_transform.clicked.connect(self.save_transform)
        self.btn_fiber_alignment.clicked.connect(self.run_fiber_alignment)
        # lambda: otherwise the button's 'checked' value (False) is passed as save_result
        self.btn_calibrate_axis.clicked.connect(lambda _checked=False: self.run_axis_calibration())

        # Path / GCode tab
        self.btn_add_waypoint.clicked.connect(self.add_waypoint)
        self.btn_clear_waypoints.clicked.connect(self.clear_waypoints)
        self.btn_run_path.clicked.connect(self.run_path)
        self.btn_save_path.clicked.connect(self.save_path)
        self.run_gcode_button.clicked.connect(self.run_gcode_from_file)

        # Camera tab
        self.btn_save_screenshot.clicked.connect(self.save_screenshot)
        self.btn_dark_shot_compensation.clicked.connect(self.capture_dark_image)

        # Home button
        self.btn_home.clicked.connect(self.home)

        self.main_tabs.setCurrentIndex(0)
        self.update_waypoint_info()

    def create_position_display(self):
        """
        Position display at the top of the control panel: live target and measured
        position, in workspace and in device (machine) coordinates, in µm.
        """
        display = QFrame(self.device_controls_frame)
        display.setObjectName("position_display")
        grid = QGridLayout(display)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)

        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        small_style = "font-size: 10pt;"
        value_style = 'font-size: 10pt; font-family: "Consolas", "Courier New", monospace;'

        self.position_title_label = QLabel("Position [µm]", display)
        grid.addWidget(self.position_title_label, 0, 0, 1, 2)
        for col, axis in enumerate(("X", "Y", "Z"), start=2):
            axis_label = QLabel(axis, display)
            axis_label.setAlignment(right)
            grid.addWidget(axis_label, 0, col)

        # (key, frame text, kind text) - frame text only on the first row of each block
        rows = [
            ("workspace_target", "Workspace", "target"),
            ("workspace_measured", "", "measured"),
            ("machine_target", "Machine", "target"),
            ("machine_measured", "", "measured"),
        ]
        self.position_value_labels = {}
        for row, (key, frame_text, kind_text) in enumerate(rows, start=1):
            for col, text in ((0, frame_text), (1, kind_text)):
                label = QLabel(text, display)
                label.setStyleSheet(small_style)
                grid.addWidget(label, row, col)

            labels = []
            for col in range(2, 5):
                value_label = QLabel("–", display)
                value_label.setAlignment(right)
                value_label.setStyleSheet(value_style)
                value_label.ensurePolished()
                # fixed width so the layout does not jump when the number of digits changes
                value_label.setMinimumWidth(value_label.fontMetrics().horizontalAdvance("-00000.0") + 4)
                grid.addWidget(value_label, row, col)
                labels.append(value_label)
            self.position_value_labels[key] = labels

        grid.setColumnStretch(1, 1)

        # first item of the control panel, above the "Micro-Manipulator Controls" heading
        self.device_controls_layout.insertWidget(0, display)
        self.position_display = display

    def set_position_values(self, key, position):
        for label, value in zip(self.position_value_labels[key], position or (None,) * 3):
            label.setText("–" if value is None else f"{value * 1000.0:.1f}")

    def clear_position_display(self, title="Position [µm]"):
        self.position_title_label.setText(title)
        for key in self.position_value_labels:
            self.set_position_values(key, None)

    def on_position_updated(self, target, measured):
        if self.position_poll_worker is None or self.sender() is not self.position_poll_worker:
            return  # late update from a worker of a previous connection

        self.set_position_values("machine_target", target)
        self.set_position_values("machine_measured", measured)
        self.set_position_values("workspace_target",
                                 self.oms.machine_to_workspace(target) if target else None)
        self.set_position_values("workspace_measured",
                                 self.oms.machine_to_workspace(measured) if measured else None)

    def on_position_unsupported(self):
        if self.position_poll_worker is None or self.sender() is not self.position_poll_worker:
            return
        self.clear_position_display("Position: needs firmware with M54")

    def start_position_polling(self):
        self.stop_position_polling()
        self.clear_position_display()
        worker = PositionPollWorker(self.oms, interval_s=0.2, parent=self)
        worker.position_updated.connect(self.on_position_updated)
        worker.unsupported.connect(self.on_position_unsupported)
        worker.finished.connect(worker.deleteLater)
        self.position_poll_worker = worker
        worker.start()

    def stop_position_polling(self):
        """Asks the polling thread to stop (without waiting) and returns it, or None."""
        worker = self.position_poll_worker
        self.position_poll_worker = None
        if worker is not None:
            worker.stop()
        return worker

    def register_stage_widget(self, widget):
        self.stage_dependent_widgets.append(widget)
        return widget

    def register_camera_widget(self, widget):
        self.camera_dependent_widgets.append(widget)
        return widget

    def show_placeholder_frame(self):
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        self.video_viewer.set_image(placeholder, pixel_per_mm=self.pixel_per_mm)

    def clear_visual_state(self):
        self.last_frame = None
        self.draw_buffer = None
        self.image_point_tracker.reset()

    def refresh_serial_devices(self):
        selected_id = self.serial_combo.currentData().get("id") if self.serial_combo.currentData() else None
        if selected_id is None:
            selected_id = self.settings.value("Connections/last_stage_id")
        self.serial_devices = list_serial_devices()
        self.serial_combo.clear()

        for device in self.serial_devices:
            self.serial_combo.addItem(device["label"], device)

        if selected_id is not None:
            for index, device in enumerate(self.serial_devices):
                if device["id"] == selected_id:
                    self.serial_combo.setCurrentIndex(index)
                    break

        self.update_connection_state()

    def refresh_camera_devices(self):
        selected_id = self.camera_combo.currentData().get("id") if self.camera_combo.currentData() else None
        if selected_id is None:
            selected_id = self.settings.value("Connections/last_camera_id")
        self.camera_devices = list_camera_devices()
        self.camera_combo.clear()

        for device in self.camera_devices:
            self.camera_combo.addItem(device["label"], device)

        if selected_id is not None:
            for index, device in enumerate(self.camera_devices):
                if device["id"] == selected_id:
                    self.camera_combo.setCurrentIndex(index)
                    break

        self.update_connection_state()

    def update_connection_state(self):
        stage_connected = self.oms.is_connected()
        camera_connected = self.camera is not None and self.camera.is_connected()
        stage_busy = self.stage_task is not None

        for widget in self.stage_dependent_widgets:
            widget.setEnabled(stage_connected and not stage_busy)

        for widget in self.camera_dependent_widgets:
            widget.setEnabled(camera_connected)

        self.serial_connect_button.setText("Close" if stage_connected else "Open")
        self.serial_connect_button.setEnabled((stage_connected or self.serial_combo.count() > 0) and not stage_busy)
        self.camera_connect_button.setText("Close" if camera_connected else "Open")
        self.camera_connect_button.setEnabled(camera_connected or self.camera_combo.count() > 0)

        stage_status = f"[{self.connected_stage_label}]" if stage_connected else "[Disconnected]"
        camera_status = f"[{self.connected_camera_label}]" if camera_connected else "[Disconnected]"
        self.serial_status_label.setText(stage_status)
        self.camera_status_label.setText(camera_status)

        for label, connected in (
            (self.serial_status_label, stage_connected),
            (self.camera_status_label, camera_connected),
        ):
            label.setProperty("connected", connected)
            label.style().unpolish(label)
            label.style().polish(label)

    def toggle_stage_connection(self):
        if self.oms.is_connected():
            self.disconnect_stage()
        else:
            self.connect_selected_stage()

    def toggle_camera_connection(self):
        if self.camera is not None and self.camera.is_connected():
            self.disconnect_camera()
        else:
            self.connect_selected_camera()

    def connect_selected_stage(self):
        if self.oms.is_connected() or self.stage_task is not None:
            return

        device = self.serial_combo.currentData()
        if device is None:
            QMessageBox.warning(self, "No Serial Device", "No serial device is available.")
            return

        port = device["port"]

        def on_done(connected, exc):
            if exc is not None or not connected:
                message = f"Failed to connect to serial device:\n{port}"
                if exc is not None:
                    message += f"\n\n{exc}"
                else:
                    message += "\n\nThe device did not answer like the manipulator firmware. Is this the right port?"
                QMessageBox.critical(self, "Connection Failed", message)
                self.connected_stage_label = None
                self.update_connection_state()
                return

            self.connected_stage_label = device["label"]
            self.settings.setValue("Connections/last_stage_id", device["id"])
            self.on_stage_connected()

        # Opening the port and checking the firmware version runs in the background: on a
        # wrong port this can take several seconds and must not freeze the window.
        self.start_stage_task(lambda: self.oms.connect(port), on_done)

    def on_stage_connected(self):
        position = self.oms.read_current_position(True)
        if position[0] is not None:
            self.current_pos = list(position)

        self.apply_acceleration_setting()
        for tool_idx, _, _ in self.tool_controls:
            self.apply_tool_setting(tool_idx)
        self.update_connection_state()
        self.start_position_polling()

    def disconnect_stage(self):
        if self.realtime_control_widget.is_running():
            self.realtime_control_widget.stop_control()

        self.stop_gcode_runner()
        poll_worker = self.stop_position_polling()
        self.oms.disconnect()  # also wakes up a position request that is still waiting
        if poll_worker is not None:
            poll_worker.wait(1000)
        self.clear_position_display()
        self.connected_stage_label = None
        self.current_pos = [0, 0, 0]
        self.update_connection_state()

    def create_camera_from_config(self, config):
        if config["kind"] == "opencv":
            return OpenCVCamera(
                camera_index=config["index"],
                backend=config.get("backend", cv2.CAP_ANY),
                resolution=(1920, 1080)
            )

        if config["kind"] == "basler":
            return BaslerCamera(device_serial=config["id"])

        raise ValueError(f"Unsupported camera kind: {config['kind']}")

    def connect_selected_camera(self):
        if self.camera is not None and self.camera.is_connected():
            return

        config = self.camera_combo.currentData()
        if config is None:
            QMessageBox.warning(self, "No Camera", "No camera is available.")
            return

        try:
            camera = self.create_camera_from_config(config)
        except Exception as exc:
            QMessageBox.critical(self, "Camera Error", str(exc))
            return

        if camera is None or not camera.is_connected():
            if camera is not None:
                camera.close()
            QMessageBox.critical(self, "Camera Error", f"Failed to connect to camera:\n{config['label']}")
            return

        self.disconnect_camera(show_placeholder=False)
        self.camera = camera
        self.connected_camera_label = config["label"]
        self.settings.setValue("Connections/last_camera_id", config["id"])
        self.clear_visual_state()
        self.load_camera_settings(self.connected_camera_label)
        self.apply_camera_settings()
        self.start_camera_stream()
        self.update_connection_state()

    def camera_settings_group(self, camera_name):
        return f"CameraSettings/{camera_name}"

    def load_camera_settings(self, camera_name):
        if not camera_name:
            return

        group = self.camera_settings_group(camera_name)
        for spinbox in (self.exposure_spinbox, self.gain_spinbox, self.white_balance_spinbox):
            spinbox.blockSignals(True)

        try:
            self.exposure_spinbox.setValue(
                self.settings.value(f"{group}/exposure", self.exposure_spinbox.value(), type=float))
            self.gain_spinbox.setValue(
                self.settings.value(f"{group}/gain", self.gain_spinbox.value(), type=float))
            self.white_balance_spinbox.setValue(
                self.settings.value(f"{group}/white_balance", self.white_balance_spinbox.value(), type=float))
        finally:
            for spinbox in (self.exposure_spinbox, self.gain_spinbox, self.white_balance_spinbox):
                spinbox.blockSignals(False)

    def save_camera_settings(self, camera_name):
        if not camera_name:
            return

        group = self.camera_settings_group(camera_name)
        self.settings.setValue(f"{group}/exposure", self.exposure_spinbox.value())
        self.settings.setValue(f"{group}/gain", self.gain_spinbox.value())
        self.settings.setValue(f"{group}/white_balance", self.white_balance_spinbox.value())

    def apply_camera_settings(self):
        if self.camera is None or not self.camera.is_connected():
            return

        try:
            self.camera.set_exposure_time(self.exposure_spinbox.value())
        except Exception:
            pass

        try:
            self.camera.set_gain(self.gain_spinbox.value())
        except Exception:
            pass

        try:
            self.camera.set_white_balance(self.white_balance_spinbox.value())
        except Exception:
            pass

        self.save_camera_settings(self.connected_camera_label)

    def start_camera_stream(self):
        if self.camera is None or not self.camera.is_connected():
            return

        self.stop_camera_stream()
        self.camera_stream_worker = CameraStreamWorker(self.camera, self)
        self.camera_stream_worker.frame_ready.connect(self.on_frame_received)
        self.camera_stream_worker.stream_error.connect(self.on_camera_stream_error)
        self.camera_stream_worker.start()

    def stop_camera_stream(self):
        if self.camera_stream_worker is not None:
            self.camera_stream_worker.stop()
            self.camera_stream_worker = None

    def disconnect_camera(self, show_placeholder=True):
        self.stop_camera_stream()

        if self.camera is not None:
            self.camera.close()
            self.camera = None

        self.connected_camera_label = None
        self.clear_visual_state()
        if show_placeholder:
            self.show_placeholder_frame()
        self.update_connection_state()

    def on_camera_stream_error(self, message):
        self.disconnect_camera()
        QMessageBox.critical(self, "Camera Stream Error", message)

    def on_frame_received(self, frame):
        try:
            if frame is None:
                return

            if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                vis_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                vis_img = frame.copy()

            self.update_controller(frame, vis_img, pixel_per_mm=self.pixel_per_mm)
        finally:
            # allow the camera thread to deliver the next frame
            if self.camera_stream_worker is not None:
                self.camera_stream_worker.frame_consumed()

    def stop_gcode_runner(self):
        if self.gcode_runner is not None:
            self.gcode_runner.stop()
            self.gcode_runner = None

        self.run_gcode_button.blockSignals(True)
        self.run_gcode_button.setChecked(False)
        self.run_gcode_button.blockSignals(False)

    def apply_acceleration_setting(self):
        if self.oms.is_connected():
            self.oms.set_max_acceleration(self.accel_spinbox.value(), 5000)

    def apply_tool_setting(self, tool_idx):
        # A tool outputs its spinbox value while its on/off button is checked, else 0.
        if not self.oms.is_connected():
            return

        _, spinbox, button = self.tool_controls[tool_idx]
        value = spinbox.value() if button.isChecked() else 0.0
        self.oms.set_tool_output(tool_idx, value, immediate=True)

    def require_stage_connection(self):
        if not self.oms.is_connected():
            QMessageBox.warning(self, "Manipulator Disconnected", "Connect a serial device before using this control.")
            return False

        # While a background task or the G-code runner is talking to the device, a command
        # from the GUI thread would block the whole window until they are done.
        if self.stage_task is not None or self.gcode_runner is not None:
            print("Manipulator is busy, command ignored.")
            return False

        return True

    def start_stage_task(self, function, on_done):
        """
        Runs 'function' (blocking device communication) in a background thread so the
        window stays responsive. 'on_done(result, exception)' is called in the GUI thread.
        Stage controls are disabled while the task runs.
        """
        task = BackgroundTask(function, self)
        task.on_done = on_done
        # connected to a method of this window, so it is executed in the GUI thread
        task.task_done.connect(self.on_stage_task_done)
        self.stage_task = task
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.update_connection_state()
        task.start()

    def on_stage_task_done(self, result, exc):
        task = self.stage_task
        self.stage_task = None
        QApplication.restoreOverrideCursor()
        self.update_connection_state()
        if task is not None:
            task.wait()
            task.deleteLater()
            task.on_done(result, exc)

    def require_camera_connection(self):
        if self.camera is not None and self.camera.is_connected():
            return True

        QMessageBox.warning(self, "Camera Disconnected", "Connect a camera before using this control.")
        return False

    def run_fiber_alignment(self):
        if not self.require_stage_connection() or not self.require_camera_connection():
            return

        # imported here because it loads scipy / scikit-optimize, which slows down startup
        from optical_alignment import OpticalAlignment

        aligner = OpticalAlignment(self.oms, self.camera)
        pos, _ = aligner.optimize()
        self.camera.start_grabbing(False)
        self.current_pos = pos

    def run_axis_calibration(self, save_result=True):
        if not self.require_stage_connection():
            return

        num_joints = 3

        # runs in a background thread (takes up to 30 s per joint)
        def calibrate_all_joints():
            joint_data = []
            for joint_index in range(num_joints):
                # save_result=True persists the calibration on the controller.
                res, data = self.oms.calibrate_joint(joint_index, save_result=save_result)
                if data and len(data) >= 3 and len(data[0]) > 0:
                    joint_data.append((joint_index, data))
            return joint_data

        def on_done(joint_data, exc):
            if exc is not None:
                QMessageBox.critical(self, "Calibration Failed", f"An error occurred during calibration:\n{exc}")
                return

            if not joint_data:
                QMessageBox.warning(self, "Calibration Failed", "No calibration data was returned by the controller.")
                return

            dialog = CalibrationPlotDialog(joint_data, self)
            dialog.show()

        self.start_stage_task(calibrate_all_joints, on_done)

    def on_stop_realtime_control(self):
        if self.oms.is_connected():
            position = self.oms.read_current_position(True)
            if position[0] is not None:
                self.current_pos[:] = position

    def home(self):
        if not self.require_stage_connection():
            return

        home_return_feedrate = 2.0  # mm/s, for the move from the end stops to zero

        # runs in a background thread (homing can take several seconds)
        def home_and_move_to_zero():
            res = self.oms.home()
            if res != SerialInterface.ReplyStatus.OK:
                raise RuntimeError(f"The device did not finish homing (reply: {res.name}).")

            # Move from the end stops to machine zero (the neutral centre of the device),
            # independent of Set Origin. move_to() expects workspace coordinates.
            zero = self.oms.machine_to_workspace((0.0, 0.0, 0.0))
            self.oms.move_to(*zero, home_return_feedrate, timeout=60)
            self.oms.wait_for_stop()
            return self.oms.read_current_position(True)

        def on_done(position, exc):
            if exc is not None:
                QMessageBox.critical(self, "Homing Failed", f"An error occurred during homing:\n{exc}")
                return

            if position is not None and position[0] is not None:
                self.current_pos = list(position)

        self.start_stage_task(home_and_move_to_zero, on_done)

    def sync_current_pos(self):
        """
        Takes over the position the device is commanded to (end of all queued moves, in
        workspace coordinates), so jogging and waypoints continue from where G-code, paths
        or mouse control left the manipulator instead of from an outdated GUI value.
        """
        try:
            position = self.oms.read_current_position(True)
        except ValueError:
            return
        if position[0] is not None:
            self.current_pos = list(position)

    def move_axis(self, axis, direction):
        if not self.require_stage_connection():
            return

        self.sync_current_pos()

        flipped = (1, 1, 1)
        d = self.step_sizes[self.step_size_idx]
        self.current_pos[axis] += direction * d * flipped[axis]
        self.current_pos[axis] = max(min(self.current_pos[axis], 10), -10)
        self.oms.move_to(*self.current_pos, self.feedrates[self.step_size_idx])

    def add_waypoint(self):
        if not self.require_stage_connection():
            return

        if self.realtime_control_widget.is_running():
            self.current_pos[:] = self.realtime_control_widget.get_current_pose()
        else:
            self.sync_current_pos()

        self.waypoints.append([self.current_pos.copy(), self.feedrates[self.step_size_idx]])
        self.update_waypoint_info()

    def run_path(self):
        if not self.require_stage_connection():
            return

        if len(self.waypoints) == 0:
            return

        # Previously the path was advanced by incoming camera frames, so it only ran with
        # a camera connected. Now all waypoints (and the return to the first one) are sent
        # from a background thread. The device queues them and executes them in order.
        waypoints = [(list(pos), feedrate) for pos, feedrate in self.waypoints]
        waypoints.append(waypoints[0])

        def send_path():
            for pos, feedrate in waypoints:
                self.oms.move_to(*pos, feedrate, timeout=60)
                self.oms.dwell(0.1, True, timeout=60)
            return waypoints[-1][0]

        def on_done(last_pos, exc):
            if exc is not None:
                QMessageBox.critical(self, "Path Failed", f"An error occurred while sending the path:\n{exc}")
                return
            self.current_pos = list(last_pos)

        self.start_stage_task(send_path, on_done)

    def save_path(self):
        if len(self.waypoints) <= 0:
            QMessageBox.critical(self, "Save Error", "Waypoint list is empty")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Path to G-code File",
            "",
            "G-code Files (*.g *.gcode);;Text Files (*.txt);;All Files (*.*)",
        )

        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("; Generated by Open Micro Manipulator Controller\n")
                handle.write("G90 ; absolute positioning\n\n")
                # waypoints are in mm, the G-code runner assumes micrometers without this
                handle.write("SCALE=1\n")

                for waypoint in self.waypoints:
                    (x, y, z), feedrate = waypoint
                    handle.write(f"G0 X{x:.10f} Y{y:.10f} Z{z:.10f} F{feedrate * 60:.3f}\n")
                    handle.write("G4 S0.100000\n")

                handle.write("\n; End of file\n")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save G-code file:\n{exc}")

    def set_origin(self):
        if not self.require_stage_connection():
            return

        position = self.oms.read_current_position(True)
        if position[0] is None:
            QMessageBox.critical(self, "Read Error", "Failed to read current stage position.")
            return

        self.current_pos[:] = position
        transform = self.oms.get_workspace_transform()
        transform[0, 3] += self.current_pos[0]
        transform[1, 3] += self.current_pos[1]
        transform[2, 3] += self.current_pos[2]
        self.oms.set_workspace_transform(transform)
        self.current_pos = [0, 0, 0]
        self.oms.move_to(0, 0, 0, self.feedrates[self.step_size_idx])

    def clear_waypoints(self):
        self.waypoints.clear()
        self.update_waypoint_info()

    def update_waypoint_info(self):
        self.waypoint_info_label.setText(f"Path Control  [ {len(self.waypoints)} Waypoints ]")

    def set_step_size(self, index):
        self.step_size_idx = index

    def update_controller(self, frame, vis_image, pixel_per_mm):
        prev_pos = self.image_point_tracker.prev_pos
        current_pos = self.image_point_tracker.update(frame)

        if current_pos is not None:
            px, py = current_pos
            cv2.circle(vis_image, (px, py), 4, (255, 0, 0), thickness=-1)

            if self.draw_buffer is None:
                self.draw_buffer = np.zeros_like(vis_image)
            elif prev_pos is not None:
                cv2.line(self.draw_buffer, prev_pos, current_pos, color=(255, 255, 255), thickness=1)

            if self.draw_buffer is not None:
                mask = self.draw_buffer[:, :, 0] != 0
                vis_image[mask, :] = np.array((0, 255, 100), dtype=np.uint8)

        self.video_viewer.set_image(vis_image, pixel_per_mm=pixel_per_mm)
        self.last_frame = frame
        return vis_image

    def set_tracking_point(self):
        if not self.require_camera_connection():
            return

        if self.last_frame is None:
            return

        self.image_point_tracker.set_track_point(
            self.last_frame,
            self.last_frame.shape[1] // 2,
            self.last_frame.shape[0] // 2,
        )

    def clear_draw_buffer(self):
        if self.draw_buffer is not None:
            self.draw_buffer = None
            self.image_point_tracker.reset()

    def run_gcode_from_file(self, checked):
        if checked:
            if not self.require_stage_connection():
                self.run_gcode_button.blockSignals(True)
                self.run_gcode_button.setChecked(False)
                self.run_gcode_button.blockSignals(False)
                return

            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open G-code File",
                "",
                "G-code Files (*.g *.gcode);;Text Files (*.txt);;All Files (*.*)",
            )

            if not path:
                self.run_gcode_button.blockSignals(True)
                self.run_gcode_button.setChecked(False)
                self.run_gcode_button.blockSignals(False)
                return

            try:
                with open(path, "r", encoding="utf-8") as handle:
                    gcode = handle.read()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to open file:\n{exc}")
                self.run_gcode_button.blockSignals(True)
                self.run_gcode_button.setChecked(False)
                self.run_gcode_button.blockSignals(False)
                return

            # max_feedrate in mm/s (was 0.5, which made every G-code run very slow)
            self.gcode_runner = GCodeRunner(gcode, self.oms, max_feedrate=5.0)

            def on_finished():
                # called from the runner thread: widgets must only be touched in the
                # GUI thread, so hand over via a signal
                self.gcode_finished_signal.emit()

            def on_iteration_finished():
                pass

            self.gcode_runner.progress_updated.connect(
                lambda fraction: self.gcode_progress_bar.setValue(int(fraction * 100))
            )

            self.gcode_runner.run(on_finished,
                                  on_iteration_finished,
                                  loop_playback=False,
                                  tool_power=self.tool1_spinbox.value())
            
        elif self.gcode_runner is not None:
            self.gcode_runner.stop()

    def on_gcode_runner_finished(self):
        self.gcode_runner = None
        self.run_gcode_button.blockSignals(True)
        self.run_gcode_button.setChecked(False)
        self.run_gcode_button.blockSignals(False)

    def run_3point_alignment(self):
        if not self.require_stage_connection():
            return

        if len(self.waypoints) != 3:
            QMessageBox.critical(self, "Error", "Exactly 3 Waypoints need to be recorded for this function")
            return

        old_workspace_transform = self.oms.get_workspace_transform()

        p0 = np.array(self.waypoints[0][0])
        p1 = np.array(self.waypoints[1][0])
        p2 = np.array(self.waypoints[2][0])

        v1 = p1 - p0
        v2 = p2 - p0
        z_axis = np.cross(v1, v2)
        if z_axis[2] < 0.0:
            z_axis = -z_axis
        z_axis /= np.linalg.norm(z_axis)

        global_x = np.array([1.0, 0.0, 0.0])
        x_proj = global_x - np.dot(global_x, z_axis) * z_axis
        x_axis = x_proj / np.linalg.norm(x_proj)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = z_axis
        transform[:3, 3] = p0

        self.oms.set_workspace_transform(transform @ old_workspace_transform)
        QMessageBox.information(self, "Alignment Complete", "3-point alignment complete.")

        self.oms.move_to(0, 0, 0, self.feedrates[self.step_size_idx])
        self.current_pos = [0, 0, 0]

    def load_transform(self):
        if not self.require_stage_connection():
            return

        try:
            with open("transform.json", "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load transform:\n{exc}")
            return

        transform = np.array(data)
        self.oms.set_workspace_transform(transform)
        self.oms.move_to(0, 0, 0, self.feedrates[self.step_size_idx])
        self.current_pos = [0, 0, 0]

    def save_transform(self, pressed=True, ask=True):
        if ask:
            confirmed = QMessageBox.question(
                None,
                "Save Transform",
                "Are you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes
            if not confirmed:
                return

        try:
            transform = self.oms.get_workspace_transform()
            with open("transform.json", "w", encoding="utf-8") as handle:
                json.dump(transform.tolist(), handle)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save transform:\n{exc}")

    def closeEvent(self, event: QCloseEvent):
        if self.stage_task is not None:
            QMessageBox.information(self, "Manipulator Busy",
                                    "Please wait until homing / calibration / path upload has finished.")
            event.ignore()
            return

        if self.realtime_control_widget.is_running():
            self.realtime_control_widget.stop_control()

        self.stop_gcode_runner()
        self.disconnect_camera()
        self.disconnect_stage()
        super().closeEvent(event)

    def capture_dark_image(self):
        if not self.require_camera_connection():
            return
        self.camera.capture_dark_image()

    def save_screenshot(self):
        if self.last_frame is None:
            return

        os.makedirs("./screenshots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screenshot_{timestamp}.png"
        path = os.path.join("./screenshots", filename)
        img = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    def run_spacekey_command(self):
        if not self.require_stage_connection():
            return

        self.realtime_control_widget.stop_control()
        pos = self.oms.read_current_position(True)

        mode = 1
        if mode == -1:
            self.oms.set_tool_output(0, self.tool1_spinbox.value(), False)
            self.oms.dwell(0.1, False)
            self.oms.set_tool_output(0, 0.00, False)
            self.oms.dwell(0.1, False)
            return
        if mode == 1: # row
            num_cycles = 1
            on_step = 0.05
            off_step = 0.00
            accel_step = 0.001
            decel_step = 0.001
            feed = 0.05

            y = pos[1] + accel_step
            self.oms.move_to(pos[0], y, pos[2], feed)
            for _ in range(num_cycles):
                self.oms.set_tool_output(0, self.tool1_spinbox.value(), False)
                y += on_step
                self.oms.move_to(pos[0], y, pos[2], feed)
                self.oms.set_tool_output(0, 0.00, False)
                y += off_step
                self.oms.move_to(pos[0], y, pos[2], feed)
            y += decel_step
            self.oms.move_to(pos[0], y, pos[2], feed)
            self.oms.move_to(pos[0], pos[1], pos[2]-0.001, 10)
            self.oms.dwell(0.1, True)
        if mode == 2:
            side = 0.05
            feed = 0.1

            self.oms.set_tool_output(0, self.tool1_spinbox.value(), False)
            self.oms.move_to(pos[0]-side, pos[1], pos[2], feed)
            self.oms.move_to(pos[0]-side, pos[1]+side, pos[2], feed)
            self.oms.move_to(pos[0], pos[1]+side, pos[2], feed)
            self.oms.move_to(pos[0], pos[1], pos[2], feed)
            self.oms.move_to(pos[0], pos[1], pos[2]-0.003, feed)
            self.oms.set_tool_output(0, 0.0, False)
            self.oms.dwell(0.1, True)
        if mode == 3:
            num_lines = 10
            spacing = 0.01
            length = 0.3
            feed = 0.1
            up = pos[2] - 0.01

            self.oms.set_tool_output(0, self.tool1_spinbox.value(), False)
            for i in range(num_lines):
                x = pos[0] + i * spacing
                if i > 0:
                    self.oms.move_to(x, pos[1], up, feed*10)
                self.oms.move_to(x, pos[1], pos[2], feed*10)
                self.oms.set_tool_output(0, self.tool1_spinbox.value(), False)
                self.oms.move_to(x, pos[1] + length, pos[2], feed)
                self.oms.set_tool_output(0, 0.0, False)

            self.oms.move_to(pos[0], pos[1], pos[2], feed)
            self.oms.move_to(pos[0], pos[1], pos[2]+0.003, feed)
            self.oms.set_tool_output(0, 0.0, False)
            self.oms.dwell(0.1, True)
        if mode == 4:
            on_time_s = self.tool1_spinbox.value()
            self.oms.set_tool_output(0, 1.0, False)
            self.oms.dwell(on_time_s, True)
            self.oms.set_tool_output(0, 0.0, False)
            self.oms.dwell(0.5, True)

        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_A:
            self.move_axis(0, -1)
        elif event.key() == Qt.Key.Key_D:
            self.move_axis(0, +1)
        elif event.key() == Qt.Key.Key_W:
            self.move_axis(1, -1)
        elif event.key() == Qt.Key.Key_S:
            self.move_axis(1, +1)
        elif event.key() == Qt.Key.Key_R:
            self.move_axis(2, +1)
        elif event.key() == Qt.Key.Key_F:
            self.move_axis(2, -1)
        elif event.key() == Qt.Key.Key_K:
            self.add_waypoint()
        elif event.key() == Qt.Key.Key_P:
            self.save_screenshot()
        elif event.key() == Qt.Key.Key_F11:
            self.toolbar_widget.setVisible(not self.toolbar_widget.isVisible())
        elif event.key() == Qt.Key.Key_Space:
            self.run_spacekey_command()
        else:
            super().keyPressEvent(event)
