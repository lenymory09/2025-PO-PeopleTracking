# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QSizePolicy, QVBoxLayout, QWidget)

class CameraLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: black;")
        self.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        # Force un ratio 16:9

        w = event.size().width()
        h = int(w * 9 / 16)

        # Si la hauteur dépasse, on ajuste avec la hauteur
        if h > event.size().height():
            h = event.size().height()
            w = int(h * 16 / 9)

        self.resize(w, h)
        super().resizeEvent(event)

class Ui_PersonTracker(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"PersonTracker")
        #main_window.resize(1919, 1075)
        # sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(main_window.sizePolicy().hasHeightForWidth())
        # main_window.setSizePolicy(sizePolicy)
        main_window.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonFollowStyle)
        self.centralwidget = QWidget(main_window)
        self.centralwidget.setObjectName(u"centralwidget")
        # sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # sizePolicy1.setHorizontalStretch(0)
        # sizePolicy1.setVerticalStretch(0)
        # sizePolicy1.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        # self.centralwidget.setSizePolicy(sizePolicy1)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cameras_layout = QVBoxLayout()
        self.cameras_layout.setObjectName(u"cameras_layout")
        self.camera_3 = CameraLabel(self.centralwidget)
        self.camera_3.setObjectName(u"camera_3")
        # sizePolicy1.setHeightForWidth(self.camera_3.sizePolicy().hasHeightForWidth())
        # self.camera_3.setSizePolicy(sizePolicy1)
        #self.camera_3.setScaledContents(True)

        self.cameras_layout.addWidget(self.camera_3)

        self.camera_4 = CameraLabel(self.centralwidget)
        self.camera_4.setObjectName(u"camera_4")
        # sizePolicy1.setHeightForWidth(self.camera_4.sizePolicy().hasHeightForWidth())
        #self.camera_4.setSizePolicy(sizePolicy1)
        #self.camera_4.setScaledContents(True)

        self.cameras_layout.addWidget(self.camera_4)


        self.horizontalLayout.addLayout(self.cameras_layout)

        self.verticalWidget = QWidget(self.centralwidget)
        self.verticalWidget.setObjectName(u"verticalWidget")
        self.verticalLayout = QVBoxLayout(self.verticalWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label = QLabel(self.verticalWidget)
        self.label.setObjectName(u"label")
        # sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        # self.label.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignHCenter)

        self.verticalLayout_2.addWidget(self.label)

        self.nombres_personnes_label = QLabel(self.verticalWidget)
        self.nombres_personnes_label.setObjectName(u"nombres_personnes_label")
        # sizePolicy1.setHeightForWidth(self.nombres_personnes_label.sizePolicy().hasHeightForWidth())
        # self.nombres_personnes_label.setSizePolicy(sizePolicy1)
        font1 = QFont()
        font1.setPointSize(75)
        font1.setBold(False)
        font1.setKerning(True)
        self.nombres_personnes_label.setFont(font1)
        self.nombres_personnes_label.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.verticalLayout_2.addWidget(self.nombres_personnes_label)


        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.logs_personnes = QLabel(self.verticalWidget)
        self.logs_personnes.setObjectName(u"logs_personnes")
        # sizePolicy1.setHeightForWidth(self.logs_personnes.sizePolicy().hasHeightForWidth())
        # self.logs_personnes.setSizePolicy(sizePolicy1)
        font2 = QFont()
        font2.setPointSize(15)
        self.logs_personnes.setFont(font2)
        self.logs_personnes.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.logs_personnes.setWordWrap(False)

        self.verticalLayout.addWidget(self.logs_personnes)


        self.horizontalLayout.addWidget(self.verticalWidget)
        self.horizontalLayout.setStretch(0, 1)  # cameras_layout
        self.horizontalLayout.setStretch(1, 1)  # verticalWidget

        main_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(main_window)

        QMetaObject.connectSlotsByName(main_window)
    # setupUi

    def retranslateUi(self, PersonTracker):
        PersonTracker.setWindowTitle(QCoreApplication.translate("PersonTracker", u"PersonTracker", None))
        self.camera_3.setText(QCoreApplication.translate("PersonTracker", u"TextLabel", None))
        self.camera_4.setText(QCoreApplication.translate("PersonTracker", u"TextLabel", None))
        self.label.setText(QCoreApplication.translate("PersonTracker", u"Nombres de personnes estim\u00e9s :", None))
        self.nombres_personnes_label.setText(QCoreApplication.translate("PersonTracker", u"0", None))
        self.logs_personnes.setText(QCoreApplication.translate("PersonTracker", u"TextLabel", None))
    # retranslateUi

