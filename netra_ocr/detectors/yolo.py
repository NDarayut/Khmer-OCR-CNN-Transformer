import os
import cv2
from PIL import Image
from .base import BaseTextDetector, DetectedLine

_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "yolo26s_best", "best.pt"
)


class YoloDetector(BaseTextDetector):
    def __init__(self, weights: str = _WEIGHTS_PATH, conf: float = 0.25, iou: float = 0.7,
                 pad: int = 2):
        from ultralytics import YOLO
        print(f"Loading YOLO weights: {weights}")
        self._model = YOLO(weights)
        self._conf = conf
        self._iou  = iou
        self._pad  = max(0, int(pad))

    def detect(self, image_path: str) -> list:
        results = self._model.predict(
            source=image_path,
            conf=self._conf,
            iou=self._iou,
            imgsz=640,
            verbose=False,
        )
        img_bgr = cv2.imread(image_path)
        img_h, img_w = img_bgr.shape[:2]

        detected = []
        for result in results:
            boxes   = result.boxes.xyxy.cpu().numpy()   # (N, 4) — x1 y1 x2 y2
            classes = result.boxes.cls.cpu().numpy()    # (N,)  integer class ids
            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = (
                    max(0, int(box[0]) - self._pad), max(0, int(box[1]) - self._pad),
                    min(img_w, int(box[2]) + self._pad), min(img_h, int(box[3]) + self._pad),
                )
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop_bgr = img_bgr[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                label = "logo" if int(cls_id) == 1 else "text"
                detected.append(DetectedLine(bbox=(x1, y1, x2, y2), crop=pil_crop, label=label))

        return detected
