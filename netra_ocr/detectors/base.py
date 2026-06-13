from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image


@dataclass
class DetectedLine:
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in original image pixels
    crop: Image.Image                  # PIL RGB crop ready for recognize_batch
    label: str = "text"               # "text" | "logo"


class BaseTextDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> List[DetectedLine]:
        """Return detected text lines for the given image file."""
        ...
