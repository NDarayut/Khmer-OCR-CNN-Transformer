from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class DetectionLevel(Enum):
    CHARACTER = "character"
    WORD = "word"
    LINE = "line"
    BLOCK = "block"


@dataclass
class TextBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    level: DetectionLevel = DetectionLevel.LINE
    children: List["TextBox"] = field(default_factory=list)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
