from .tesseract import TesseractDetector
from .yolo import YoloDetector
from .legacy import LegacyDetector

DETECTOR_REGISTRY = {
    "tesseract": TesseractDetector,
    "yolo":      YoloDetector,
    "legacy":    LegacyDetector,
}


def get_detector(name: str, **kwargs):
    cls = DETECTOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {list(DETECTOR_REGISTRY)}"
        )
    return cls(**kwargs)
