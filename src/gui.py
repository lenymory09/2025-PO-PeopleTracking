import sys
from typing import List

import cv2
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QImage, QPixmap
import cv2
import math

from person_tracker import PersonTracker, Camera


class GUIApplication(QtWidgets.QWidget):
    def __init__(self, person_tracker: PersonTracker):
        super().__init__()
        self.person_tracker = person_tracker
        self.cameras: List[Camera] = person_tracker.cameras
        self.labels = []
        self.layout = QtWidgets.QGridLayout(self)

        for camera in person_tracker.cameras:
            cam_widget = QtWidgets.QWidget()
            cam_layout = QtWidgets.QVBoxLayout()
            cam_widget.setLayout(cam_layout)

            img_label = QtWidgets.QLabel()
            img_label.setFixedSize(1280, 768)
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
            frame = cam.track_people(self.person_tracker.known_embeddings)

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

    sys.exit(app.exec())
