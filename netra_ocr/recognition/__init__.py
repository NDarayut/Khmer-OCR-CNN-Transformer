"""Standalone Khmer text-line recognizer.

This subpackage has no dependency on `netra_ocr.detectors` or the full
`KhmerOCRPipeline` -- it only needs a text-line image (a file path or a PIL
Image, already cropped to one line) and returns the recognized string. Use
it directly if you already have line crops from your own detector/pipeline
and don't need `netra_ocr`'s bundled detectors at all:

    from netra_ocr.recognition import recognize, recognize_batch

    text = recognize("line_crop.png")
    texts = recognize_batch([crop1, crop2, crop3])  # PIL Images or paths
"""

from .recognize_text import recognize, recognize_batch

__all__ = ["recognize", "recognize_batch"]
