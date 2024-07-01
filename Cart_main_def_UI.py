import sys
import math
import os
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem, QImage, QPixmap
from PIL import ImageGrab
import cv2
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def numpyQImage(self,nimage):

    qImg = QImage()
    if nimage.dtype == np.uint8:
        if len(np.shape(nimage)) == 2:
            channels = 1
            height = np.shape(nimage)[0]
            width = np.shape(nimage)[1]
            bytesPerLine = channels * width
            ximage = nimage.copy()
            #qImg = QtGui.QImage(ximage, width, height, QtGui.QImage.Format_Indexed8)
            qImg = QImage(ximage.data, width, height, bytesPerLine, QImage.Format_Indexed8)
            #np.frombuffer(image.data,dtype=np.uint8)
            qImg.setColorTable([qRgb(i, i, i) for i in range(256)])
        elif len(nimage.shape) == 3:
            if nimage.shape[2] == 3:
                #print('nimg=', len(np.shape(nimage)))
                height, width, channels = nimage.shape
                #print('h,w,c = ', height, width, channels)
                #print(nimage.data)
                bytesPerLine = channels * width
                qImg = QImage(nimage.data.tobytes(), width, height, bytesPerLine, QImage.Format_RGB888)
            elif nimage.shape[2] == 4:
                height, width, channels = nimage.shape
                bytesPerLine = channels * width
                fmt = QImage.Format_ARGB32
                qImg = QImage(nimage.data, width, height, bytesPerLine, QImage.Format_ARGB32)

    return qImg

def disp_label_image(self, widget, image_data):
    #self.np_img_data = image_data
    nqimg = self.numpyQImage(image_data)
    pqimg = QPixmap(nqimg)
    nqimg = pqimg.scaled(pqimg.width(), pqimg.height())
    widget.resize(pqimg.width(), pqimg.height())
    widget.setPixmap(nqimg)
    self.show()


def resize_img(self, coloredImg, im_size, convert_br):
    imh = coloredImg.shape[0]
    imw = coloredImg.shape[1]

    im_w = im_size
    im_h = int(im_size * imh / imw)

    if convert_br == 1:
        color_rgb = cv2.cvtColor(coloredImg, cv2.COLOR_BGR2RGB)
    else:
        color_rgb = coloredImg

    resize_img = cv2.resize(color_rgb, (im_w, im_h), interpolation=cv2.INTER_AREA)

    return resize_img


def slot_open_single(self):
    fname = QFileDialog.getOpenFileName(self, 'Open cartialge file', self.pre_set_dir, filter='Image files (*.jpg *.tif *.png)')

    if fname[0]:

        self.cartilage_list_file = fname[0][:-4] + '.lbl'

        self.open_dir_name = os.path.dirname(fname[0])
        self.pre_set_dir = self.open_dir_name +'/'
        with open('./preset_dir.env', 'w') as wf:
            wf.write(self.pre_set_dir)

        self.cartilage_file_name = fname[0]
        img_array = np.fromfile(fname[0], np.uint8)
        coloredImg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        self.color_rgb = self.resize_img(coloredImg, 700, 1)
        self.disp_label_image(self.lbl_cartilage_image, self.color_rgb)


def slot_open_folder(self):
    self.open_dir_name = str(QFileDialog.getExistingDirectory(self, "Select cartilage file folder", self.pre_set_dir))

    if self.open_dir_name:
        #print('open folder', self.open_dir_name)
        self.pre_set_dir = self.open_dir_name + '/'
        with open('./preset_dir.env', 'w') as wf:
            wf.write(self.pre_set_dir)

        self.setWindowTitle(self.open_dir_name)

        file_list = os.listdir(self.open_dir_name)

        self.model1.removeRows(0, self.model1.rowCount())

        for file in file_list:
            if '.jpg' in file or '.png' in file or '.tif' in file:
                self.model1.appendRow(QStandardItem(file))

        self.listView.setModel(self.model1)

def slot_select_list(self, index):
    self.cartilage_file_name = self.open_dir_name + '/' + index.data()
    self.cartilage_list_file = self.open_dir_name + '/' + index.data()[:-4] + '.lbl'
    self.setWindowTitle(self.cartilage_file_name)

    img_array = np.fromfile(self.cartilage_file_name, np.uint8)
    coloredImg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    self.org_image = coloredImg

    self.color_rgb = self.resize_img(coloredImg, 700, 1)

    self.f_pad_image = 1

    #self.disp_label_image(self.lbl_cartilage_image, self.color_rgb)
    self.slot_draw_main_imge()

    if self.cBoxAutoPrediction.isChecked():
        self.auto_prediction()

    #cv2.imwrite('./cur_image.jpg', cv2.cvtColor(self.color_rgb, cv2.COLOR_RGB2BGR))

def angle_changed(self, angle):
    #print(angle)
    crot = float(angle)
    rgb_image = cv2.cvtColor(self.org_image, cv2.COLOR_BGR2RGB)

    # NumPy 배열을 PIL 이미지로 변환합니다.
    image = Image.fromarray(rgb_image)

    org_img_rot = image.rotate(crot)
    org_img_rot_array = np.asarray(org_img_rot)

    self.resize_rot_img = self.resize_img(org_img_rot_array, 700, 0)

    #self.disp_label_image(self.lbl_cartilage_image, self.resize_rot_img)
    self.color_rgb = self.resize_rot_img

    self.slot_draw_main_imge()



def sel_draw_circle(self, id, rd, color):

    if id == 0:
        csize = int(rd * 2)
        center = (rd, rd)
        ndata = np.ones([csize, csize, 3], dtype=np.uint8)
        ndata *= 255
        ndata = cv2.circle(ndata, center, rd, color, 2)
        ndata = cv2.circle(ndata, center, 2, (255, 0, 0), 2)
        odata_a = cv2.cvtColor(ndata, cv2.COLOR_BGR2BGRA)
        odata_a[:, :, 3] = np.zeros([csize, csize])
        odata_a[:, :, 3] = 255 - odata_a[:, :, 0]  # /255

    elif id == 1:
        csize = int(rd * 2)
        center = (rd, rd)
        ndata = np.ones([csize, csize, 3], dtype=np.uint8)
        ndata *= 255
        ndata = cv2.circle(ndata, center, rd - 1, color, 2)
        ndata = cv2.circle(ndata, center, 3, (0, 255, 255), 2)
        odata_a = cv2.cvtColor(ndata, cv2.COLOR_BGR2BGRA)
        odata_a[:, :, 3] = np.zeros([csize, csize])
        odata_a[:, :, 3] = 255 - odata_a[:, :, 0]  # /255

    return odata_a


def slot_draw_main_imge(self):

    self.p1_p_cx = int(self.p1_cx) + 130
    self.p1_p_cy = int(self.p1_cy) + 10
    self.p2_p_cx = int(self.p2_cx) + 130
    self.p2_p_cy = int(self.p2_cy) + 10

    self.bx0 = int(self.p1_p_cx)
    self.by0 = int(self.p1_p_cy)
    self.bx1 = int(self.p2_p_cx)
    self.by1 = int(self.p2_p_cy)

    if self.bx0 < 0: self.bx0 = 0
    if self.by0 < 0: self.by0 = 0

    #bx0_txt = 'box x1: %d' % (self.bx0)
    #bx1_txt = 'box x2: %d' % (self.bx1)
    #by0_txt = 'box y1: %d' % (self.by0)
    #by1_txt = 'box y2: %d' % (self.by1)

    #self.lbl_box_x0.setText(bx0_txt)
    #self.lbl_box_y0.setText(by0_txt)
    #self.lbl_box_x1.setText(bx1_txt)
    #self.lbl_box_y1.setText(by1_txt)

    cradius = 10

    self.disp_label_image(self.lbl_line_p1, self.sel_draw_circle(0, cradius, (0, 0, 0)))
    self.disp_label_image(self.lbl_line_p2, self.sel_draw_circle(0, cradius, (0, 0, 0)))

    ix = self.lbl_cartilage_image.x()
    iy = self.lbl_cartilage_image.y()

    dimg = self.color_rgb.copy()

    #print(ix, iy)
    #print(self.p1_p_cx, self.p1_p_cy)

    self.lbl_line_p1.move(ix + self.p1_p_cx - cradius, iy + self.p1_p_cy - cradius)
    self.lbl_line_p2.move(ix + self.p2_p_cx - cradius, iy + self.p2_p_cy - cradius)

    dimg = cv2.rectangle(dimg, (self.p1_p_cx, self.p1_p_cy), (self.p2_p_cx, self.p2_p_cy), (0, 0, 255), lineType=cv2.LINE_AA)

    #self.color_rgb = dimg
    self.disp_label_image(self.lbl_cartilage_image, dimg)

    return



def slot_list_save(self):
    pass


def slot_list_reset(self):
    pass

