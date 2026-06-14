# ocr_engine.py
"""
Khmer OCR Pipeline — pluggable text detector, plain-text output.

Detection
---------
Detectors are registered in netra_ocr/detectors/__init__.py and selected by name.
Currently available:
    yolo       (default) — YOLOv2.6s trained on Khmer documents (class 0: text, class 1: logo)
    tesseract            — KhmerLineDetector built on Tesseract + graph clustering

Add a new detector by:
  1. Implementing BaseTextDetector in netra_ocr/detectors/<name>.py
  2. Adding it to DETECTOR_REGISTRY in detectors/__init__.py

Output formats (auto-detected from extension)
---------------------------------------------
  .txt  .md    — plain UTF-8 text (text lines only)
  .json         — structured metadata (image size, per-line text + bbox)
  .docx         — Word document: text lines as paragraphs, logos as inline images
"""

import os
import shutil
import argparse
import sys
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from detectors import get_detector
from recognition.recognize_text import recognize_batch
from output_formatters import save_output, SUPPORTED_FORMATS


class KhmerOCRPipeline:
    def __init__(self, detector: str = "yolo", conf: float | None = None,
                 pad: int | None = None):
        print(f"Initializing detector: {detector}")
        kwargs = {}
        if conf is not None:
            kwargs["conf"] = conf
        if pad is not None:
            kwargs["pad"] = pad
        self.detector = get_detector(detector, **kwargs)

    def process_image(
        self,
        image_path: str,
        output_path: str | None = None,
        save_debug: bool = False,
        beam_width: int = 1,
        batch_size: int = 8,
        return_segments: bool = False,
    ):
        """Run detection + recognition on an image.

        Returns the joined text by default. When ``return_segments=True`` returns
        ``(text, meta)`` where ``meta`` carries the original image size and a
        JSON-serializable list of per-region segments (bbox + text/label) for the
        web overlay.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        image_size = img.size

        def _empty_meta():
            return {"image_size": [image_size[0], image_size[1]], "segments": []}

        # STEP 1: DETECTION
        detected_lines = self.detector.detect(image_path)
        if not detected_lines:
            return ("", _empty_meta()) if return_segments else ""

        # STEP 2: SEPARATE BY LABEL
        text_lines = [dl for dl in detected_lines if dl.label == "text"]
        logo_lines  = [dl for dl in detected_lines if dl.label == "logo"]

        # STEP 3: RECOGNITION (text only)
        ocr_queue = [dl.crop for dl in text_lines]
        recognitions = recognize_batch(ocr_queue, beam_width=beam_width, batch_size=batch_size) \
                       if ocr_queue else []

        # STEP 4: BUILD SEGMENTS (merged, sorted top-to-bottom)
        segments = []
        for dl, text in zip(text_lines, recognitions):
            segments.append({"type": "text", "text": text, "bbox": dl.bbox})
        for dl in logo_lines:
            segments.append({"type": "logo", "crop": dl.crop, "bbox": dl.bbox})
        segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))

        # STEP 5: SAVE
        if save_debug:
            self._save_debug(image_path, segments)

        final_text = "\n".join(s["text"] for s in segments if s["type"] == "text")

        if output_path:
            save_output(segments, output_path, image_path=image_path, image_size=image_size)

        if return_segments:
            overlay_segments = []
            for s in segments:
                x1, y1, x2, y2 = s["bbox"]
                item = {"type": s["type"], "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                if s["type"] == "text":
                    item["text"] = s["text"]
                overlay_segments.append(item)
            meta = {"image_size": [image_size[0], image_size[1]], "segments": overlay_segments}
            return final_text, meta

        return final_text

    def _save_debug(self, image_path: str, segments: list) -> None:
        base   = os.path.splitext(os.path.basename(image_path))[0]
        folder = f"debug_{base}"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        for i, seg in enumerate(segments):
            if seg["type"] == "text":
                with open(os.path.join(folder, f"text_{i:03d}.txt"), "w", encoding="utf-8") as fh:
                    fh.write(seg["text"])
            else:
                seg["crop"].save(os.path.join(folder, f"logo_{i:03d}.png"))

        n_text = sum(1 for s in segments if s["type"] == "text")
        n_logo = sum(1 for s in segments if s["type"] == "logo")
        print(f"  [Debug] {n_text} text + {n_logo} logo segments saved to '{folder}/'")


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Khmer OCR — pluggable detector pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Supported output formats: {', '.join(sorted(SUPPORTED_FORMATS))}\n\n"
            "Examples:\n"
            "  python ocr_engine.py --image scan.jpg --output result.txt\n"
            "  python ocr_engine.py --image scan.jpg --output result.docx --detector yolo\n"
            "  python ocr_engine.py --image scan.jpg --output result.json --detector yolo --conf 0.4\n"
        ),
    )
    parser.add_argument("--image",      required=True)
    parser.add_argument("--detector",   default="yolo",
                        help="Text detector: yolo | tesseract | legacy")
    parser.add_argument("--output",     default="ocr_result.txt",
                        help="Extension determines format: .txt .md .json .docx")
    parser.add_argument("--conf",       type=float, default=None,
                        help="YOLO confidence threshold (default 0.25). Only applies to --detector yolo.")
    parser.add_argument("--pad",        type=int, default=None,
                        help="Pixels to pad each bbox on all sides (default 4). Only applies to --detector yolo.")
    parser.add_argument("--beam",       type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    try:
        pipeline = KhmerOCRPipeline(detector=args.detector, conf=args.conf, pad=args.pad)
        pipeline.process_image(
            image_path  = args.image,
            output_path = args.output,
            save_debug  = args.debug,
            beam_width  = args.beam,
            batch_size  = args.batch_size,
        )
    except Exception as exc:
        print(f"\nPipeline error: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
