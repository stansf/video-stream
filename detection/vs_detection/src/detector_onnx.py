import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .constants import MODEL_PATH_ONNX
from .utils import NAMES_TESI as NAMES, COLORS_TESI as COLORS


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def add_visualization(image: np.ndarray, detections: List[Dict]):
    for detection in detections:
        label = detection['label']
        bbox = detection['bbox']
        name, _ = label.split(':')
        name = name.strip()
        color = COLORS[name]
        cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
        cv2.putText(image, label, (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    [225, 255, 255], thickness=2)
    return image


class Detector:
    def __init__(self: 'Detector', weights: Path, use_cuda=False):
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if use_cuda else ['CPUExecutionProvider'])

        self.session = ort.InferenceSession(str(weights), providers=providers)

    def _postprocess_outputs(
            self, outputs: np.ndarray, dwdh: int, ratio: float
    ) -> List[Dict]:
        results = []
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = NAMES[cls_id]
            name += ': ' + str(score)
            results.append(dict(label=name, bbox=box))
        return results

    def forward(self, img: np.ndarray) -> List[Dict]:
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)

        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        outputs = self.session.run(outname, inp)[0]
        return self._postprocess_outputs(outputs, dwdh, ratio)

    def forward_vis(self, img: np.ndarray) -> np.ndarray:
        detections = self.forward(img)
        add_visualization(img, detections)
        return img


@lru_cache()
def get_detector():
    weights_path = os.getenv('MODEL_PATH_ONNX', MODEL_PATH_ONNX)
    d = Detector(weights_path, use_cuda=False)
    return d


if __name__ == '__main__':
    from pprint import pprint

    d = Detector(Path('yolov7-tiny.onnx'))
    img = cv2.imread('./image.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = d.forward(img)
    pprint(results)
