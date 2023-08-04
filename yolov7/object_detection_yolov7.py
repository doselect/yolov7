import os
import sys
import pathlib

import copy
import torch
import numpy as np

# provide path to models
module_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(module_path)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from constants import ObjectDetectionConstants

class Yolov7():
    image_size = None
    device = None
    convert_image = None
    model = None
    objects_to_be_detected = []

    def __init__(self, objects_to_be_detected=None):
        self.image_size = ObjectDetectionConstants.OBJECT_DETECTION_IMAGE_SIZE.value
        self.device = select_device(ObjectDetectionConstants.OBJECT_DETECTION_PROCESSOR.value)
        self.convert_image = self.device.type != ObjectDetectionConstants.OBJECT_DETECTION_PROCESSOR.value
        model_path = module_path.replace(os.getcwd() + "/", "")
        self.load_model_weights()
        self.model = attempt_load([model_path + f'/{ObjectDetectionConstants.OBJECT_DETECTION_MODEL_NAME.value}'], 
                                  map_location=self.device)
        if objects_to_be_detected is None:
            objects_to_be_detected = copy.deepcopy(ObjectDetectionConstants.OBJECT_DETECTION_OBJECTS.value)
        self.objects_to_be_detected = objects_to_be_detected
        self.stride = int(self.model.stride.max())  # model stride
    
    def load_model_weights(self):
        file = "/home/sajalrastogi/DoSelect/VideoProctoring/do-proctor-engine-api/src/yolov7.pt"
        file = pathlib.Path(str(file).strip().replace("'", '').lower())
        if not file.exists():
            os.system(f'curl -L {ObjectDetectionConstants.OBJECT_DETECTION_WIEGHTS_PATH.value}')

    def load_image(self, image):
        img = letterbox(image, self.image_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img
    
    def detect_object(self, img):
        # dataset = LoadImages(image_path, img_size=self.image_size, stride=self.stride)
        # for path, img, im0s, vid_cap in dataset:
        #     pass
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            prediction = self.model(img, augment=ObjectDetectionConstants.OBJECT_DETECTION_OPT.value['augment'])[0]
        # Apply NMS
        prediction = non_max_suppression(prediction, ObjectDetectionConstants.OBJECT_DETECTION_OPT.value['conf_thres'], 
                                   ObjectDetectionConstants.OBJECT_DETECTION_OPT.value['iou_thres'], 
                                   classes=ObjectDetectionConstants.OBJECT_DETECTION_OPT.value['classes'], 
                                   agnostic=ObjectDetectionConstants.OBJECT_DETECTION_OPT.value['agnostic_nms'])[0]
        prediction = prediction.tolist()
        
        for rect in prediction:
            rect[-1] = self.model.names[int(rect[-1])]
            for i in range(4):
                rect[i] = int(rect[i])

        for i, item in enumerate(prediction):
            label = item[-1]
            if label in list(self.objects_to_be_detected.keys()):
                # item list containes boundary box till index 4 and index 4 is confidence
                self.objects_to_be_detected[label].append({'bbox': item[:4], 'confidence': item[4]})
        # return boxes, confidences, labels
        return self.objects_to_be_detected

