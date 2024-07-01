import os, sys
import torch

from PyQt5 import QtWidgets
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from Cartilage_torch_menu import Ui_MainWindow
import numpy as np

class Form(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self.setupUi(self)

        if os.path.isfile('./preset_dir.env'):
            with open('./preset_dir.env', 'r') as rf:
                self.pre_set_dir = rf.readline().strip('\n')
        else:
            with open('./preset_dir.env','w') as wf:
                self.pre_set_dir = './'
                wf.write(self.pre_set_dir)


        self.f_P1_Mou_Left = 0
        self.f_P2_Mou_Left = 0

        self.pre_p1_x = 0
        self.pre_p1_y = 0
        self.pre_p2_x = 0
        self.pre_p1_y = 0

        self.p1_cx = self.bx0 = 150
        self.p1_cy = self.by0 = 40
        self.p2_cx = self.bx1 = 340
        self.p2_cy = self.by1 = 150


        self.f_pad_image = 0

        self.pre_set_pano_dir = './'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.statusBar()

        temp_cartilage_image = np.ones((400,700,3)).astype(np.uint8)*100
        self.disp_label_image(self.lbl_cartilage_image, temp_cartilage_image)

        self.model1 = QStandardItemModel()

        self.yolo_init()
        print('yolo_v7 init!')
        self.vgg_angle_model()
        print('angle model init!')
        self.vgg_classification_model()
        print('classification model init!')

        self.setWindowTitle('Evaluate Cartilage OA')

        self.show()


    from Cart_main_def_UI import (
        slot_open_single,
        slot_open_folder,
        slot_select_list,
        angle_changed,
        resize_img,
        slot_list_save,
        slot_list_reset,
        numpyQImage,
        disp_label_image,
        sel_draw_circle,
        slot_draw_main_imge
    )

    from DL_models import (
        yolo_crop_predict,
        yolo_init,
        vgg_angle_model,
        run_angle_image,
        vgg_classification_model,
        prepare_image,
        run_predict_class,
        auto_prediction,
        slot_all_prediction
    )

    from MouseEvent import (
        eventFilter, Event_lblDispP1_Pos, Event_lblDispP2_Pos
    )


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # QtCore.QTimer.singleShot(0, self.close())
            # self.close()
            event.accept()
            # print('Window closed')
            QtWidgets.QApplication.exit()
        else:
            event.ignore()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Form()
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec())

