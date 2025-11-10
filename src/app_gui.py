# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLayout,
    QMainWindow, QSizePolicy, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1919, 1075)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy1)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.cameras_layout = QVBoxLayout()
        self.cameras_layout.setObjectName(u"cameras_layout")
        self.camera_3 = QLabel(self.centralwidget)
        self.camera_3.setObjectName(u"camera_3")
        sizePolicy1.setHeightForWidth(self.camera_3.sizePolicy().hasHeightForWidth())
        self.camera_3.setSizePolicy(sizePolicy1)
        self.camera_3.setScaledContents(True)

        self.cameras_layout.addWidget(self.camera_3)

        self.camera_4 = QLabel(self.centralwidget)
        self.camera_4.setObjectName(u"camera_4")
        sizePolicy1.setHeightForWidth(self.camera_4.sizePolicy().hasHeightForWidth())
        self.camera_4.setSizePolicy(sizePolicy1)
        self.camera_4.setScaledContents(True)

        self.cameras_layout.addWidget(self.camera_4)


        self.horizontalLayout.addLayout(self.cameras_layout)

        self.stats_layout_2 = QVBoxLayout()
        self.stats_layout_2.setObjectName(u"stats_layout_2")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignHCenter)

        self.stats_layout_2.addWidget(self.label)

        self.nombres_personnes_label = QLabel(self.centralwidget)
        self.nombres_personnes_label.setObjectName(u"nombres_personnes_label")
        sizePolicy1.setHeightForWidth(self.nombres_personnes_label.sizePolicy().hasHeightForWidth())
        self.nombres_personnes_label.setSizePolicy(sizePolicy1)
        font1 = QFont()
        font1.setPointSize(75)
        font1.setBold(False)
        font1.setKerning(True)
        self.nombres_personnes_label.setFont(font1)
        self.nombres_personnes_label.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.stats_layout_2.addWidget(self.nombres_personnes_label)


        self.horizontalLayout.addLayout(self.stats_layout_2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.camera_3.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.camera_4.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Nombres de personnes estim\u00e9s :", None))
        self.nombres_personnes_label.setText(QCoreApplication.translate("MainWindow", u"0", None))
    # retranslateUi

