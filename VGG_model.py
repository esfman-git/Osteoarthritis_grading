import os, sys
#import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#import torch.utils.data as data
#import torchvision
#import torchvision.transforms as transforms
#import torchvision.datasets as Datasets
#import matplotlib.pyplot as plt
#from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#test_path = 'd:/AI_Data/JHRuy/torch/last_rot_crop_2_cat8/test/05'
#img_name = test_path + '/AIBP_006.jpg'
#model_path = 'd:/AI_Data/JHRuy/torch/last_rot_crop_2_cat8/results/vgg16_state_s320_sgd_cat8_ep50.pt'
#OUTPUT_DIM = 8
img_size = 320

class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()        
        self.features = features        
        self.avgpool = nn.AdaptiveAvgPool2d(7)        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
                512, 512, 512, 512, 'M']

def get_vgg_layers(config, batch_norm):    
    layers = []
    in_channels = 3
    
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = c
            
    return nn.Sequential(*layers)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_predictions(model, iterator):
    model.eval()
    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred, _ = model(x)
            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    return images, labels, probs


class GradCam(nn.Module):
    def __init__(self, model, module, layer):
        super().__init__()
        self.model = model
        self.module = module
        self.layer = layer
        self.register_hooks()

    def register_hooks(self):
        for modue_name, module in self.model._modules.items():
            if modue_name == self.module:
                for layer_name, module in module._modules.items():
                    if layer_name == self.layer:
                        module.register_forward_hook(self.forward_hook)
                        module.register_backward_hook(self.backward_hook)

    def forward(self, input, target_index):
        outs, _ = self.model(input)
        #print(outs)
        outs = outs.squeeze()  # [1, num_classes]  --> [num_classes]

        # 가장 큰 값을 가지는 것을 target index 로 사용
        if target_index is None:
            target_index = outs.argmax()

        outs[target_index].backward(retain_graph=True)
        a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)  # [512, 1, 1]
        out = torch.sum(a_k * self.forward_result, dim=0).cpu()  # [512, 7, 7] * [512, 1, 1]
        out = torch.relu(out) / torch.max(out)  # 음수를 없애고, 0 ~ 1 로 scaling # [7, 7]
        out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [img_size, img_size])  # 4D로 바꿈
        return out.cpu().detach().squeeze().numpy()

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])


def show_cam_on_image(img, mask):
    img = img.resize((img_size, img_size))
    cvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #np_img = np.array(img)
    print(np.size(cvImage))
    #sys.exit()
    # Convert RGB to BGR
    #cv_img = np_img[:, :, ::-1].copy()
    #print(np.size(cv_img))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(cvImage) / 255
    cam = cam / np.max(cam)
    cv2.imshow("cam", np.uint8(255 * cam))
    cv2.imshow("heatmap", np.uint8(heatmap * 255))
    cv2.waitKey()


def vgg16_model(OUTPUT_DIM):
    vgg16_layers = get_vgg_layers(vgg16_config, batch_norm = True)
    model = VGG(vgg16_layers, OUTPUT_DIM)
    return model

def preprocess_image(img):
    preprocessed_img = img.copy()[:, :, ::-1]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input