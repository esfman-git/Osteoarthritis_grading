import os,sys
import math

from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.torch_utils import TracedModel

from VGG_model import vgg16_model, GradCam, preprocess_image

from PyQt5.QtGui import QPixmap, QImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

save_txt = False
save_conf = True
augment = True
agnostic_nms = True
classes = None
iou_thres = 0.5
conf_thres = 0.25
save_img = True
view_img = False

def yolo_crop_predict(self):

    self.img_size = 640
    half = self.device.type != 'cpu'

    im0s = cv2.cvtColor(self.color_rgb, cv2.COLOR_RGB2BGR)
    img = letterbox(im0s, self.img_size, stride=self.stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    print(np.shape(img))
    print(np.shape(im0s))

    org_h = im0s.shape[0]
    org_w = im0s.shape[1]
    yolo_h = img.shape[1]
    yolo_w = img.shape[2]

    img = torch.from_numpy(img).to(self.device)
    img = img.half() if half else img.float()

    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print(img.size())

    # Inference
    with torch.no_grad():
        pred = self.y_model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections
    prop = 0.0
    sel_i = 0
    for i, det in enumerate(pred):
        if len(det):
            num_val = det.cpu().numpy()
            if num_val[i,4] > prop:
                num_val = num_val
                prop = num_val[i,4]
                sel_i = i
        else:
            sel_i = -1

    if sel_i>=0:
        img0 = im0s
        img0 = img0.astype(np.uint8)
        self.bx0 = int(num_val[sel_i,0] * (org_w / yolo_w))
        self.bx1 = int(num_val[sel_i,2] * (org_w / yolo_w))
        self.by0 = int(num_val[sel_i,1] * (org_h / yolo_h))
        self.by1 = int(num_val[sel_i,3] * (org_h / yolo_h))

        self.color_rgb = self.resize_img(img0, 700, 1)

        #bx0_txt = 'box x1: %d' % (self.bx0)
        #bx1_txt = 'box x2: %d' % (self.bx1)
        #by0_txt = 'box y1: %d' % (self.by0)
        #by1_txt = 'box y2: %d' % (self.by1)

        #self.lbl_box_x0.setText(bx0_txt)
        #self.lbl_box_y0.setText(by0_txt)
        #self.lbl_box_x1.setText(bx1_txt)
        #self.lbl_box_y1.setText(by1_txt)

        self.p1_cx = self.bx0 - 130
        self.p1_cy = self.by0 - 10
        self.p2_cx = self.bx1 - 130
        self.p2_cy = self.by1 - 10

        #if self.p1_cy < 0 : self.p1_cy = 0

        self.slot_draw_main_imge()

def yolo_init(self):
    # Initialize
    set_logging()
    half = self.device.type != 'cpu'
    self.imgsz = 640

    # Load model
    self.y_model = attempt_load('weight_yolov7_best.pt', map_location=self.device)
    self.stride = int(self.y_model.stride.max())
    self.imgsz = check_img_size(self.imgsz, s=self.stride)

    #if trace:
    self.y_model = TracedModel(self.y_model, self.device, self.imgsz)

    if half:
        self.y_model.half()

    self.y_model.eval()


def vgg_angle_model(self):
    self.angle_model = vgg16_model(OUTPUT_DIM=1)
    self.angle_model.to(self.device)
    self.angle_model.load_state_dict(torch.load('vgg16_state_s300_2023_1_angle_ep100.pt'))
    self.angle_model.eval()


def run_angle_image(self):

    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

    print(self.cartilage_file_name)
    rgb_image = cv2.cvtColor(self.org_image, cv2.COLOR_BGR2RGB)

    # Covert numpy array to PIL image
    image = Image.fromarray(rgb_image)
    rw, rh = image.size

    # Convert the image to PyTorch tensor
    img_tensor = test_transform(image)

    img_tensor.unsqueeze_(0)
    x = img_tensor.to(self.device)
    y_pred, _ = self.angle_model(x)
    p_angle = y_pred.cpu().detach().numpy()[0]

    rot = float(p_angle) * -1
    crot = math.degrees(math.atan(math.tan(math.radians(rot)) * rh / rw))

    self.sBoxAngle.setValue(int(crot))


def auto_prediction(self):
    self.run_angle_image()
    #self.eval_yolo(self.cartilage_file_name)
    self.yolo_crop_predict()
    self.run_predict_class()


def prepare_image(self, oimg, img_size):
    oh = oimg.shape[0] / img_size
    ow = oimg.shape[1] / img_size
    oimg = cv2.resize(oimg, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    img = oimg.reshape(1, img_size, img_size, 3)
    img = img.astype('float32')

    img = img / 255.0

    return ow, oh, img, oimg


def vgg_classification_model(self):
    self.class_model = vgg16_model(OUTPUT_DIM=8)
    self.class_model.to(self.device)
    self.class_model.load_state_dict(torch.load('vgg16_state_s224_sgd_cat8_ep50.pt'))
    self.class_model.eval()

def run_predict_class(self):

    img_size = 224

    if (self.bx1 <= self.bx0) or (self.by1 <= self.by0):
        return

    print('Run!!!')

    h = self.by1 - self.by0
    w = self.bx1 - self.bx0
    cv_crop_img = self.color_rgb[self.by0 + 1: self.by0 + h - 1, self.bx0 + 1: self.bx0 + w - 1]
    img = Image.fromarray(cv_crop_img)

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    # Convert the image to PyTorch tensor
    img_tensor = test_transforms(img)

    img_tensor.unsqueeze_(0)
    img_x = img_tensor.to(self.device)
    y_pred, _ = self.class_model(img_x)
    y_prob = F.softmax(y_pred, dim=-1)

    y_pred_tensor = y_pred.cpu()
    y_pred_numpy = y_pred_tensor.detach().numpy()
    print(y_pred_numpy)

    top_pred = y_prob.argmax(1, keepdim=True)
    top_index = top_pred.cpu().item()

    # predict the class
    res_class = top_index
    if res_class == 0: res_class_txt = '0.0'
    if res_class == 1: res_class_txt = '0.5'
    if res_class == 2: res_class_txt = '1.0'
    if res_class == 3: res_class_txt = '2.0'
    if res_class == 4: res_class_txt = '3.0'
    if res_class == 5: res_class_txt = '4.0'
    if res_class == 6: res_class_txt = '5.0'
    if res_class == 7: res_class_txt = '6.0'

    print(res_class_txt)
    self.lbl_oa_score.setText('Score: ' + res_class_txt)

    grad_cam = GradCam(model=self.class_model, module='features', layer='43')

    gimg = np.float32(cv2.resize(cv_crop_img, (img_size, img_size))) / 255
    input = preprocess_image(gimg)
    input = input.to(self.device)
    mask = grad_cam(input, None)

    img_w = self.bx1 - self.bx0
    img_h = self.by1 - self.by0

    gimg = cv2.resize(cv_crop_img, (img_w, img_h))
    cimg = np.float32(gimg) / 255

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + cimg
    cam = cam / np.max(cam)
    cam = np.uint8(cam * 255)

    self.cam_color_rgb = self.color_rgb.copy()
    self.cam_color_rgb[self.by0:self.by0+img_h, self.bx0:self.bx0+img_w] = cam

    self.disp_label_image(self.lbl_grad_cam_image, self.cam_color_rgb)

    fig = Figure(figsize=(3, 2))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    y_pred_numpy = np.squeeze(y_pred_numpy)
    y_pred_numpy -= np.min(y_pred_numpy)
    x_class = ['0', '0.5', '1', '2', '3', '4', '5', '6']
    ax.bar(x_class, y_pred_numpy, color = 'red')
    ax.yaxis.set_visible(False)
    ax.set_xticklabels(x_class)

    canvas.draw()
    image = QImage(canvas.buffer_rgba(), canvas.get_width_height()[0], canvas.get_width_height()[1],
                   QImage.Format_ARGB32)

    pixmap = QPixmap(image)

    self.lblDispPred.setPixmap(pixmap)



def slot_all_prediction(self):
    pass

