import pathlib
import sys
module_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(module_path)
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np

opt = {
    "augment": False,
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'classes': None,
    'agnostic_nms': False,
}

class Yolov7():
    def __init__(self, classes=None): 
        self.device = select_device("")
        self.half = self.device.type != 'cpu'
        self.output_folder = os.getcwd() + "/output_images/"
        model_path = module_path.replace(os.getcwd() + "/", "")
        self.model = attempt_load([model_path + '/yolov7.pt'], map_location=self.device)
        if classes is None:
            classes = self.model.names
        self.classes = classes
        self.classes_index = [self.model.names.index(x) for x in self.classes]
        self.stride = int(self.model.stride.max())  # model stride
        return
    
    def get_pred(self, image_path, save_image=False, output_folder=None):
        t1 = time.time()
        dataset = LoadImages(image_path, img_size=640, stride=self.stride)
        for path, img, im0s, vid_cap in dataset:
            pass
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=opt['augment'])[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])[0]
        pred = pred.tolist()
        image0 = cv2.imread(image_path)
        H, W = image0.shape[:2]
        m = max(H, W)
        if save_image:
            for element in pred:
                # element[:4] = element[:4]
                element[:4] = [int(x * m/640 ) for x in element[:4]]
                rect = element[:4]
                element[-1] = int(element[-1])
                # print(rect, element[-1])
                if element[-1] in self.classes_index:
                    conf = round(element[4], 2)
                    cv2.rectangle(image0, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                    cv2.putText(image0, f"{self.model.names[element[-1]]} {conf}", (rect[0], rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if output_folder is None:
                output_folder = self.output_folder
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cv2.imwrite(output_folder + f"/{Path(image_path).name}", image0)
        for item in pred:
            # print(item)
            item.append(self.model.names[int(item[-1])])
        return pred
