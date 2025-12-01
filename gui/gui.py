import os
import sys
import threading
from typing import List, Optional

from tracking import Camera
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage, QPixmap, QFont, QFontDatabase
import cv2
import time

from reid import EnhancedReID
from .app_gui import Ui_PersonTracker
import queue
from DB import DB


class GUIApp(QtWidgets.QMainWindow, Ui_PersonTracker):
    """
    Classe de la GUI
    """

    def __init__(self, config):
        super().__init__()

        self.reid = EnhancedReID(config)
        self.cameras: List[Camera] = []
        for idx, source in enumerate(config['video']['sources']):
            self.cameras.append(Camera(source, self.reid, config, idx))

        if len(self.cameras) > 2:
            print("Il ne peut y avoir que 2 sources de caméras.")
            exit(1)
        self.setupUi(self)
        self.resize(960, 540)
        self.cameras_labels = [self.camera_1, self.camera_2]
        self.running = False
        self.processors = []
        self.frame_queues = {}
        self.update_thread: Optional[threading.Thread] = None
        self.db = DB()
        self.legend_font = QFontDatabase.font("Science Gothic", "Science Gothic Regular", 10)
        self.number_font = QFontDatabase.font("Science Gothic", "Science Gothic Regular", 10)
        self.logs_font = QFontDatabase.font("Science Gothic", "Science Gothic Regular", 10)
        self.reid.load_features()
        if not os.path.exists(os.path.join(os.getcwd(), self.reid.embeddings_path)):
            self.db.create_db()
            self.start_nb_passages = 0
        else:
            nb_passage = self.load_nb_persons()
            self.start_nb_passages = nb_passage

    def start_processing(self):
        """
        Lance le processing des caméras.
        """
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
            if i == 0:
                threading.Thread(target=cam.run, daemon=True).start()
            else:
                threading.Thread(target=cam.run_tracking, daemon=True).start()

        self.update_thread = threading.Thread(target=self.update_frames, daemon=True)
        self.update_thread.start()

        self.save_persons_timer = QtCore.QTimer(self)
        self.save_persons_timer.timeout.connect(self.save_persons_confirmed)
        self.save_persons_timer.setInterval(1000 * 5)
        self.save_persons_timer.start()

        self.update_logs_timer = QtCore.QTimer(self)
        self.update_logs_timer.timeout.connect(self.update_logs)
        self.update_logs_timer.setInterval(1000 * 2)
        self.update_logs_timer.start()

        self.save_features_timer = QtCore.QTimer(self)
        self.save_features_timer.timeout.connect(self.save_features)
        self.save_features_timer.setInterval(60000)
        self.save_features_timer.start()

    def save_features(self):
        """
        Enregistre les features dans des fichiers.
        """
        self.reid.save_features()

    def update_nombre_personnes(self):
        """
        Modifie le nombre de personne affiché
        """
        # nb_persons = self.db.fetch_nb_personnes()[0]
        if len(self.cameras) == 2:
            nb_persons = (len(self.cameras[1].passages_entrees) + self.start_nb_passages) // 2
            self.nombres_personnes_label.setText(f"∼ {nb_persons}")

    def update_logs(self):
        """
        Met à jour les logs dans la GUI
        """

        all_ids = []

        for camera in self.cameras:
            all_ids += camera.current_persons
        ids = list(set(all_ids))
        if ids:
            persons = self.db.fetch_personnes(ids)
            string = ""
            for id_person, timestamp in persons:
                string += f"ID {id_person} : {timestamp}\n"

            if string != "":
                self.logs_personnes.setText(string)

    def save_persons_confirmed(self):
        """
        Sauvegarde les personnes confirmés (qui ont un certain nombre de features) qui ne sont pas enregistrés.
        """
        persons = self.reid.get_confirmed_persons()
        self.db.insert_visites(persons)

    def update_frames(self):
        """
        Met à jour les frames des caméras dans la GUI.
        """

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

    def resizeEvent(self, event):
        """
        Gère le redimensionnement de l'application.
        :param event: évennement de redimensionnement.
        """
        w = event.size().width()

        font_size_legend = max(13, w // 60)
        self.legend_font.setPointSize(font_size_legend)
        self.label.setFont(self.legend_font)

        font_size_number = max(13, w // 10)
        self.number_font.setPointSize(font_size_number)
        self.nombres_personnes_label.setFont(self.number_font)

        font_size_logs = max(13, w // 75)
        self.logs_font.setPointSize(font_size_logs)
        self.logs_personnes.setFont(self.logs_font)

        super().resizeEvent(event)

    def release_ressources(self):
        """
        libère les ressources utilisés (caméras, db, etc.)
        """
        for camera in self.cameras:
            camera.release()

        self.db.close_db()

    def save_nombres_persons(self):
        """
        Sauvegarde du nombre de personnes dans un fichier.
        """
        with open("nb_persons.txt", "w") as f:
            f.write(str(len(self.cameras[1].passages_entrees) + self.start_nb_passages))

    @staticmethod
    def load_nb_persons():
        """
        Chargement du nombre de fichier.
        :return: Le nombre de passages devant la caméra enregistrés.
        """
        if not os.path.exists("nb_persons.txt"):
            return 0
        with open("nb_persons.txt", "r") as f:
            return int(f.read())


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    gui = GUIApp({})
    gui.show()
    sys.exit(app.exec())
