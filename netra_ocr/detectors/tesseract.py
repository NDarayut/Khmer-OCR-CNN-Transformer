import cv2
from PIL import Image
from .base import BaseTextDetector, DetectedLine


class TesseractDetector(BaseTextDetector):
    def __init__(self):
        from engine import KhmerLineDetector
        print("Initializing KhmerLineDetector (Tesseract)...")
        self._detector = KhmerLineDetector()

    def detect(self, image_path: str) -> list:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        result = self._detector.detect(img_bgr)
        lines = []
        for lb, crop_np in zip(result.lines, result.line_crops):
            pil_crop = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
            bbox = (lb.x, lb.y, lb.x + lb.w, lb.y + lb.h)
            lines.append(DetectedLine(bbox=bbox, crop=pil_crop))
        return lines
