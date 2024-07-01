# Grading of Osteoarthritis in Cartilage

![Example Image](https://github.com/esfman-git/Osteoarthritis_grading/blob/main/fiqures/program_img_capture.JPG)

We use a VGG16 regression model to adjust the angle of the entire image so that the cartilage area is displayed horizontally. The cartilage area is then cropped using YOLOv7. The YOLOv7 model used in this program is sourced from https://github.com/WongKinYiu/yolov7. A VGG16 classification model is used to grade the cartilage into 8 levels (0, 0.5, 1, 2, 3, 4, 5, 6).

This program is designed with a GUI using PYQT5. By clicking on the images listed in the left window’s list box and then clicking the buttons in the right window, you can sequentially view the cartilage grading. If the crop is incorrect, you can manually adjust the cartilage area by moving the two points in the image window.

The VGG16 regression weight values needed to horizontalize the cartilage area can be downloaded from https://drive.google.com/file/d/1vVYFrAkchgNYkATRISAqiZk590kggrlp/view?usp=sharing. 
The VGG16 weight values for Cartilage OA grading are available at https://drive.google.com/file/d/1_TOGQ6VXYP94psTj3QCx1AlquiX-8Pul/view?usp=sharing."