from enum import Enum
import os

class ObjectDetectionConstants(Enum):

    OBJECT_DETECTION_OPT = {
        'augment': False,
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'classes': None,
        'agnostic_nms': False,
        }
    
    OBJECT_DETECTION_WIEGHTS_PATH = os.environ.get('OBJECT_DETECTION_WIEGHTS', 'https://doselect-dev-packages.s3.ap-southeast-1.amazonaws.com/yolov3.pt')
    OBJECT_DETECTION_IMAGE_SIZE = 640
    OBJECT_DETECTION_PROCESSOR = 'cpu'
    OBJECT_DETECTION_MODEL_NAME = 'yolov7.pt'
    OBJECT_DETECTION_IMAGE_DIMENSIONS = 3
    OBJECT_DETECTION_OBJECTS = {'cell phone': []}
