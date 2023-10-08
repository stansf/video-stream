import time
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np
import openvino as ov
import torch

from .constants import MODEL_PATH_OV
from .utils import (preprocess_image, prepare_input_tensor, draw_boxes,
                   non_max_suppression, NAMES_TESI as NAMES,
                   COLORS_TESI as COLORS)


def detect(model: ov.Model, img: np.ndarray,
           conf_thres: float = 0.25, iou_thres: float = 0.45,
           classes: List[int] = None, agnostic_nms: bool = False):
    """
    OpenVINO YOLOv7 model inference function. Reads image, preprocess it,
    runs model inference and postprocess results using NMS.
    Parameters:
        model (ov.Model): OpenVINO compiled model.
        image_path (Path): input image path.
        conf_thres (float, *optional*, 0.25): minimal accpeted confidence for object filtering
        iou_thres (float, *optional*, 0.45): minimal overlap score for remloving objects duplicates in NMS
        classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
    Returns:
       pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
       orig_img (np.ndarray): image before preprocessing, can be used for results visualization
       inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
    """
    output_blob = model.output(0)
    preprocessed_img, orig_img = preprocess_image(img)
    input_tensor = prepare_input_tensor(preprocessed_img)

    start = time.perf_counter()
    predictions = model(input_tensor)[output_blob]
    end = time.perf_counter()

    predictions = torch.tensor(predictions)
    pred = non_max_suppression(predictions, conf_thres, iou_thres,
                               classes=classes, agnostic=agnostic_nms)
    return pred, orig_img, input_tensor.shape, end - start


class Detector:
    def __init__(self, model_path: Path):
        core = ov.Core()
        model = core.read_model(str(model_path))
        self.compiled_model = core.compile_model(model, 'CPU')

    def forward_vis(self, img: np.ndarray, with_fps: bool = False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, image, input_shape, proc_time = detect(self.compiled_model,
                                                      img, conf_thres=0.35)
        image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES,
                                      COLORS)
        if with_fps:
            fps = int(round(1 / proc_time))
            print(f'FPS: {fps:3} | Time : {proc_time}')
        return image_with_boxes


@lru_cache()
def get_detector() -> Detector:
    model_path = os.getenv('MODEL_PATH_OV', MODEL_PATH_OV)
    d = Detector(model_path)
    return d


def main():
    core = ov.Core()
    model = core.read_model('model/yolov7-tiny_int8.xml')
    compiled_model = core.compile_model(model, 'CPU')
    img = cv2.imread('image_0.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, image, input_shape = detect(compiled_model, img)
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES, COLORS)
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    cv2.imwrite('bbxs.jpg', image_with_boxes)


if __name__ == '__main__':
    main()
