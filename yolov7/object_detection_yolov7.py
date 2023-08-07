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

class Yolov7:
    """wrapper for yolov7 implementation
    """
    image_size = None
    device = None
    convert_image = None
    model = None
    objects_detected_confidence_mapping = []

    def __init__(self, objects_to_be_detected=None):
        self.image_size = ObjectDetectionConstants.OBJECT_DETECTION_IMAGE_SIZE.value
        self.device = select_device(ObjectDetectionConstants.OBJECT_DETECTION_PROCESSOR.value)
        self.convert_image = self.device.type != ObjectDetectionConstants.OBJECT_DETECTION_PROCESSOR.value
        model_root_path = module_path.replace(os.getcwd() + "/", "")
        model_path = model_root_path + f'/{ObjectDetectionConstants.OBJECT_DETECTION_MODEL_NAME.value}'
        self.load_model_weights(model_path)
        self.model = attempt_load([model_path], 
                                  map_location=self.device) 
        if objects_to_be_detected is None:
            self.objects_detected_confidence_mapping = copy.deepcopy(ObjectDetectionConstants.OBJECT_DETECTION_OBJECTS.value)
        else:
            self.objects_detected_confidence_mapping = self.get_object_confidence_mapping(objects_to_be_detected)
        self.objects_to_be_detected = objects_to_be_detected
        self.stride = int(self.model.stride.max())  # model stride
    

    def load_model_weights(self, model_path):
        """load model weights from model_path
        :param model_path: str
        """
        file = pathlib.Path(str(model_path).strip().replace("'", '').lower())
        if not file.exists():
            os.system(f'wget {ObjectDetectionConstants.OBJECT_DETECTION_WIEGHTS_PATH.value}')


    def get_object_confidence_mapping(self, objects_to_be_detected):
        """get object and confidence mapping for objects list

        :param objects_to_be_detected: list
        :return: dict
        """
        objects_detected_confidence_mapping = {}
        for _object in objects_to_be_detected:
            objects_detected_confidence_mapping[_object] = []
        return objects_detected_confidence_mapping


    def preprocess_image(self, image):
        """preprocessed image compatible for yolov7

        :param image: np.ndarray
        :return: np.ndarray
        """
        img = letterbox(image, 640, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.convert_image else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img


    def detect_object(self, img):
        """object detection method

        :param img: np.ndarray
        :return: dict: {object: [{confidence: , bbox: }]}
        """
        img = self.preprocess_image(image=img)

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
            if label in list(self.objects_detected_confidence_mapping.keys()):
                # item list containes boundary box till index 4 and index 4 is confidence
                self.objects_detected_confidence_mapping[label].append({'bbox': item[:4], 'confidence': item[4]})

        return self.objects_detected_confidence_mapping

