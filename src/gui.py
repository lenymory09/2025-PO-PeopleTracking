import sys
import threading
from typing import List, Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QImage, QPixmap
import cv2
import math
import time

from PySide6.QtWidgets import QLabel

from person_tracker import EnhancedPersonTracker as PersonTracker, Camera
from app_gui import Ui_MainWindow
import queue

class GUIApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, config):
        super().__init__()
        self.person_tracker = PersonTracker(config)
        self.cameras: List[Camera] = self.person_tracker.cameras
        self.setupUi(self)
        self.cameras_labels = self.horizontalLayoutWidget.findChildren(QLabel)
        print("labels:", self.cameras_labels)
        self.running = False
        self.processors = []
        self.frame_queues = {}
        self.update_thread: Optional[threading.Thread] = None

    def start_processing(self):
        # self.timer = QtCore.QTimer(self)
        # self.timer.timeout.connect(self.update_frames)
        # self.timer.start(30)
        self.timer_update_persons = QtCore.QTimer(self)
        self.timer_update_persons.timeout.connect(self.update_nombre_personnes)
        self.timer_update_persons.start(1)
        self.running = True

        self.processors = []

        for i, cam in enumerate(self.cameras):
            frame_queue = queue.Queue(maxsize=10)
            self.frame_queues[i] = frame_queue
            cam.frame_queue = frame_queue
            self.processors.append(cam)
            threading.Thread(target=cam.run, daemon=True).start()

        self.update_thread = threading.Thread(target=self.update_frames, daemon=True)
        self.update_thread.start()

    @QtCore.Slot()
    def update_nombre_personnes(self):
        self.nombres_personnes_label.setText(str(self.person_tracker.calc_nb_persons()))

    # @QtCore.Slot()
    # def update_frames(self):
    #     for idx, camera in enumerate(self.cameras):
    #         lbl = self.cameras_labels[idx]
    #         frame = camera.track_people()
    #         if frame is not None:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             h, w, ch = frame.shape
    #             img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
    #
    #             lbl.setPixmap(QPixmap.fromImage(img).scaled(
    #                 lbl.width(), lbl.height()
    #             ))

    def update_frames(self):
        update_interval = 0.033
        last_update = time.time()

        while self.running:
            current_time = time.time()
            if current_time - last_update >= update_interval:
                for idx, camera in enumerate(self.cameras):
                    frame = None
                    try:
                        while True:
                            _, frame = self.frame_queues[idx].get_nowait()
                    except queue.Empty:
                        pass

                    lbl = self.cameras_labels[idx]
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = frame.shape
                        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)

                        lbl.setPixmap(QPixmap.fromImage(img).scaled(
                            lbl.width(), lbl.height()
                        ))
                last_update = current_time
            else:
                time.sleep(0.001)


class GUIApplication(QtWidgets.QWidget):
    def __init__(self, config):
        super().__init__()
        self.person_tracker = PersonTracker(config)
        self.cameras: List[Camera] = self.person_tracker.cameras
        self.labels = []
        self.layout = QtWidgets.QGridLayout(self)

        for camera in self.cameras:
            cam_widget = QtWidgets.QWidget()
            cam_layout = QtWidgets.QVBoxLayout()
            cam_widget.setLayout(cam_layout)

            img_label = QtWidgets.QLabel()
            # img_label.setFixedWidth(1280)
            img_label.setFixedSize(1280, 728)
            img_label.setStyleSheet("background-color: black;")

            caption = QtWidgets.QLabel(f"Camera : {camera.source}")
            caption.setAlignment(QtCore.Qt.AlignCenter)
            cam_layout.addWidget(img_label)
            cam_layout.addWidget(caption)

            self.layout.addWidget(cam_widget)
            self.labels.append(img_label)

        self._update_grid()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

    @QtCore.Slot()
    def update_frames(self):
        for cam, lbl in zip(self.cameras, self.labels):
            frame = cam.process_frame()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)

                lbl.setPixmap(QPixmap.fromImage(img).scaled(
                    lbl.width(), lbl.height()
                ))

    def resizeEvent(self, event):
        # impose le ratio 16:9
        w = event.size().width()
        h = int(w * 9 / 16)  # calcule la hauteur
        self.resize(w, h)

    def _update_grid(self):
        # vider le layout existant
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            self.layout.removeWidget(widget)

        n = len(self.labels)
        cols = math.ceil(math.sqrt(n))  # nombre de colonnes
        rows = math.ceil(n / cols)  # nombre de lignes

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < n:
                    self.layout.addWidget(self.labels[idx], r, c)
                    idx += 1


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    gui = GUIApp({})
    gui.show()
    sys.exit(app.exec())
