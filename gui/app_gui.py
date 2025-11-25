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

class Ui_PersonTracker(object):
    def setupUi(self, PersonTracker):
        if not PersonTracker.objectName():
            PersonTracker.setObjectName(u"PersonTracker")
        PersonTracker.resize(1919, 1085)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PersonTracker.sizePolicy().hasHeightForWidth())
        PersonTracker.setSizePolicy(sizePolicy)
        PersonTracker.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonFollowStyle)
        self.centralwidget = QWidget(PersonTracker)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cameras_layout = QVBoxLayout()
        self.cameras_layout.setObjectName(u"cameras_layout")
        self.camera_1 = QLabel(self.centralwidget)
        self.camera_1.setObjectName(u"camera_1")
        sizePolicy.setHeightForWidth(self.camera_1.sizePolicy().hasHeightForWidth())
        self.camera_1.setSizePolicy(sizePolicy)
        self.camera_1.setStyleSheet(u"background: black;")
        self.camera_1.setScaledContents(True)

        self.cameras_layout.addWidget(self.camera_1)

        self.camera_2 = QLabel(self.centralwidget)
        self.camera_2.setObjectName(u"camera_2")
        self.camera_2.setEnabled(True)
        sizePolicy.setHeightForWidth(self.camera_2.sizePolicy().hasHeightForWidth())
        self.camera_2.setSizePolicy(sizePolicy)
        self.camera_2.setStyleSheet(u"background: black;")
        self.camera_2.setScaledContents(True)

        self.cameras_layout.addWidget(self.camera_2)


        self.horizontalLayout.addLayout(self.cameras_layout)

        self.verticalWidget = QWidget(self.centralwidget)
        self.verticalWidget.setObjectName(u"verticalWidget")
        self.verticalLayout = QVBoxLayout(self.verticalWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label = QLabel(self.verticalWidget)
        self.label.setObjectName(u"label")
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignHCenter)

        self.verticalLayout_2.addWidget(self.label)

        self.nombres_personnes_label = QLabel(self.verticalWidget)
        self.nombres_personnes_label.setObjectName(u"nombres_personnes_label")
        sizePolicy.setHeightForWidth(self.nombres_personnes_label.sizePolicy().hasHeightForWidth())
        self.nombres_personnes_label.setSizePolicy(sizePolicy)
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
        sizePolicy.setHeightForWidth(self.logs_personnes.sizePolicy().hasHeightForWidth())
        self.logs_personnes.setSizePolicy(sizePolicy)
        font2 = QFont()
        font2.setPointSize(15)
        self.logs_personnes.setFont(font2)
        self.logs_personnes.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.logs_personnes.setWordWrap(False)

        self.verticalLayout.addWidget(self.logs_personnes)


        self.horizontalLayout.addWidget(self.verticalWidget)

        PersonTracker.setCentralWidget(self.centralwidget)

        self.retranslateUi(PersonTracker)

        QMetaObject.connectSlotsByName(PersonTracker)
    # setupUi

    def retranslateUi(self, PersonTracker):
        PersonTracker.setWindowTitle(QCoreApplication.translate("PersonTracker", u"PersonTracker", None))
        self.camera_1.setText("")
        self.camera_2.setText("")
        self.label.setText(QCoreApplication.translate("PersonTracker", u"Nombre de personnes estim\u00e9 :", None))
        self.nombres_personnes_label.setText(QCoreApplication.translate("PersonTracker", u"0", None))
        self.logs_personnes.setText(QCoreApplication.translate("PersonTracker", u"Pas encore de visites", None))
    # retranslateUi

