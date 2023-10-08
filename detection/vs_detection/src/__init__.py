from .detector_onnx import get_detector as get_detector_onnx
from .detector_ov import get_detector


__all__ = ['get_detector', 'get_detector_onnx']
