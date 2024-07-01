# Grading of Osteoarthritis in Cartilage

VGG16 rgression model을 이용하여 전체이미지에서 cartilage 영역이 수평이 되도록 각도를 조정하여 display합니다. 
YOLOv7을 이용하여 Cartilage 영역을 crop합니다. 
본 프로그램에서 사용된 YOLOv7 모델은 Github https://github.com/WongKinYiu/yolov7 에서 가져온 모델입니다. 
VGG16 classification model을 이용하여 cartialge를 8단계(0,0.5,1,2,3,4,5,6)로 grading합니다. 

본 프로그램은 PYQT5를 이용하여 GUI로 구성되어 있습니다. 
좌측 창의 리스트 박스에 나열되어 있는 이미지를 클릭하고 오른쪽 창의 버튼들을 클릭하면 순서적으로 crtilage grading을 볼 수 있습니다. 
crop이 잘못되어 있을 경우 이미지 창의 두개의 point를 조정하여 수동으로 cartilage 영역을 crop할 수 있습니다. 
