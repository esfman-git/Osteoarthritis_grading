# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Cartilage_menu.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1291, 828)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pBtnSingleOpen = QtWidgets.QPushButton(self.centralwidget)
        self.pBtnSingleOpen.setGeometry(QtCore.QRect(20, 30, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.pBtnSingleOpen.setFont(font)
        self.pBtnSingleOpen.setObjectName("pBtnSingleOpen")
        self.pBtnFolderOpen = QtWidgets.QPushButton(self.centralwidget)
        self.pBtnFolderOpen.setGeometry(QtCore.QRect(20, 80, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.pBtnFolderOpen.setFont(font)
        self.pBtnFolderOpen.setObjectName("pBtnFolderOpen")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(20, 130, 201, 471))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.listView.setFont(font)
        self.listView.setObjectName("listView")
        self.lbl_cartilage_image = QtWidgets.QLabel(self.centralwidget)
        self.lbl_cartilage_image.setGeometry(QtCore.QRect(240, 30, 641, 361))
        self.lbl_cartilage_image.setObjectName("lbl_cartilage_image")
        self.lbl_grad_cam_image = QtWidgets.QLabel(self.centralwidget)
        self.lbl_grad_cam_image.setGeometry(QtCore.QRect(240, 420, 621, 301))
        self.lbl_grad_cam_image.setObjectName("lbl_grad_cam_image")
        self.pBtnRotate = QtWidgets.QPushButton(self.centralwidget)
        self.pBtnRotate.setGeometry(QtCore.QRect(970, 180, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.pBtnRotate.setFont(font)
        self.pBtnRotate.setObjectName("pBtnRotate")
        self.pBtnCrop = QtWidgets.QPushButton(self.centralwidget)
        self.pBtnCrop.setGeometry(QtCore.QRect(970, 300, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.pBtnCrop.setFont(font)
        self.pBtnCrop.setObjectName("pBtnCrop")
        self.pBtnRunEvaluate = QtWidgets.QPushButton(self.centralwidget)
        self.pBtnRunEvaluate.setGeometry(QtCore.QRect(970, 390, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.pBtnRunEvaluate.setFont(font)
        self.pBtnRunEvaluate.setObjectName("pBtnRunEvaluate")
        self.lbl_oa_score = QtWidgets.QLabel(self.centralwidget)
        self.lbl_oa_score.setGeometry(QtCore.QRect(990, 450, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.lbl_oa_score.setFont(font)
        self.lbl_oa_score.setObjectName("lbl_oa_score")
        self.lbl_line_p1 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_line_p1.setGeometry(QtCore.QRect(360, 140, 56, 12))
        self.lbl_line_p1.setObjectName("lbl_line_p1")
        self.lbl_line_p2 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_line_p2.setGeometry(QtCore.QRect(680, 230, 56, 12))
        self.lbl_line_p2.setObjectName("lbl_line_p2")
        self.pBtnAutoPrediction = QtWidgets.QPushButton(self.centralwidget)
        self.pBtnAutoPrediction.setGeometry(QtCore.QRect(970, 100, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.pBtnAutoPrediction.setFont(font)
        self.pBtnAutoPrediction.setObjectName("pBtnAutoPrediction")
        self.sBoxAngle = QtWidgets.QSpinBox(self.centralwidget)
        self.sBoxAngle.setGeometry(QtCore.QRect(970, 230, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.sBoxAngle.setFont(font)
        self.sBoxAngle.setAlignment(QtCore.Qt.AlignCenter)
        self.sBoxAngle.setMinimum(-90)
        self.sBoxAngle.setMaximum(90)
        self.sBoxAngle.setObjectName("sBoxAngle")
        self.cBoxAutoPrediction = QtWidgets.QCheckBox(self.centralwidget)
        self.cBoxAutoPrediction.setGeometry(QtCore.QRect(970, 50, 161, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.cBoxAutoPrediction.setFont(font)
        self.cBoxAutoPrediction.setObjectName("cBoxAutoPrediction")
        self.lblDispPred = QtWidgets.QLabel(self.centralwidget)
        self.lblDispPred.setGeometry(QtCore.QRect(980, 490, 291, 211))
        self.lblDispPred.setObjectName("lblDispPred")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1291, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pBtnSingleOpen.clicked.connect(MainWindow.slot_open_single) # type: ignore
        self.pBtnFolderOpen.clicked.connect(MainWindow.slot_open_folder) # type: ignore
        self.listView.clicked['QModelIndex'].connect(MainWindow.slot_select_list) # type: ignore
        self.pBtnCrop.clicked.connect(MainWindow.yolo_crop_predict) # type: ignore
        self.pBtnRunEvaluate.clicked.connect(MainWindow.run_predict_class) # type: ignore
        self.pBtnRotate.clicked.connect(MainWindow.run_angle_image) # type: ignore
        self.sBoxAngle.valueChanged['int'].connect(MainWindow.angle_changed) # type: ignore
        self.pBtnAutoPrediction.clicked.connect(MainWindow.auto_prediction) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pBtnSingleOpen.setText(_translate("MainWindow", "Open Single Image"))
        self.pBtnFolderOpen.setText(_translate("MainWindow", "Select Folder"))
        self.lbl_cartilage_image.setText(_translate("MainWindow", "TextLabel"))
        self.lbl_grad_cam_image.setText(_translate("MainWindow", "TextLabel"))
        self.pBtnRotate.setText(_translate("MainWindow", "Rotation"))
        self.pBtnCrop.setText(_translate("MainWindow", "Crop Cartilage"))
        self.pBtnRunEvaluate.setText(_translate("MainWindow", "Run Prediction"))
        self.lbl_oa_score.setText(_translate("MainWindow", "Score"))
        self.lbl_line_p1.setText(_translate("MainWindow", "p1"))
        self.lbl_line_p2.setText(_translate("MainWindow", "p2"))
        self.pBtnAutoPrediction.setText(_translate("MainWindow", "Prediction (Auto)"))
        self.cBoxAutoPrediction.setText(_translate("MainWindow", "Auto Prediction"))
        self.lblDispPred.setText(_translate("MainWindow", "Plot"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
