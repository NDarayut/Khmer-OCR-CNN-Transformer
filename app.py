import os
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).parent
UPLOAD_FOLDER = ROOT / "uploads"
RESULTS_FOLDER = ROOT / "results"
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.urandom(24)

# Lazy-init pipeline cache: {detector_key -> KhmerOCRPipeline}
_pipelines: dict = {}


def _get_pipeline(detector: str, conf: float | None, pad: int | None):
    import sys
    netra_path = str(ROOT / "netra_ocr")
    if netra_path not in sys.path:
        sys.path.insert(0, netra_path)
    from netra_ocr.ocr_engine import KhmerOCRPipeline

    key = (detector, conf, pad)
    if key not in _pipelines:
        _pipelines[key] = KhmerOCRPipeline(detector=detector, conf=conf, pad=pad)
    return _pipelines[key]


def _cleanup_old_files(folder: Path, max_age: int = 3600) -> None:
    now = time.time()
    for f in folder.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > max_age:
            try:
                f.unlink()
            except OSError:
                pass


def _safe_path(folder: Path, name: str) -> Path | None:
    resolved = (folder / name).resolve()
    if resolved.parent != folder.resolve():
        return None
    return resolved


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(ok=False, error="No file provided."), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify(ok=False, error="Empty filename."), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(ok=False, error=f"Unsupported file type '{ext}'. Use JPG, PNG, TIFF, or BMP."), 400

    upload_id = f"{uuid.uuid4()}{ext}"
    dest = UPLOAD_FOLDER / upload_id
    f.save(dest)

    return jsonify(ok=True, upload_id=upload_id, preview_url=f"/preview/{upload_id}")


@app.route("/preview/<upload_id>")
def preview(upload_id: str):
    path = _safe_path(UPLOAD_FOLDER, upload_id)
    if path is None or not path.exists():
        return jsonify(ok=False, error="Not found."), 404
    return send_file(path)


@app.route("/run", methods=["POST"])
def run():
    body = request.get_json(silent=True) or {}
    upload_id = body.get("upload_id", "")
    detector = body.get("detector", "yolo")
    output_format = body.get("output_format", ".txt")
    conf_raw = body.get("conf")
    pad_raw = body.get("pad")

    if detector not in ("tesseract", "yolo", "legacy"):
        return jsonify(ok=False, error=f"Unknown detector '{detector}'."), 400

    conf = float(conf_raw) if detector == "yolo" and conf_raw is not None else None
    pad = int(pad_raw) if detector in ("yolo", "legacy") and pad_raw is not None else None

    image_path = _safe_path(UPLOAD_FOLDER, upload_id)
    if image_path is None or not image_path.exists():
        return jsonify(ok=False, error="Uploaded file not found. Please re-upload."), 404

    stem = Path(upload_id).stem
    result_filename = f"{stem}_{int(time.time())}{output_format}"
    result_path = RESULTS_FOLDER / result_filename

    try:
        pipeline = _get_pipeline(detector, conf, pad)
        text, meta = pipeline.process_image(
            image_path=str(image_path),
            output_path=str(result_path),
            beam_width=1,
            batch_size=8,
            return_segments=True,
        )
    except ImportError as e:
        return jsonify(ok=False, error=f"Detector not available: {e}. Try Tesseract instead."), 500
    except FileNotFoundError as e:
        return jsonify(ok=False, error=str(e)), 404
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    finally:
        _cleanup_old_files(UPLOAD_FOLDER)
        _cleanup_old_files(RESULTS_FOLDER)

    display_text = text if text else "(No text detected in this document.)"

    return jsonify(
        ok=True,
        text=display_text,
        result_id=result_filename,
        download_url=f"/download/{result_filename}",
        image_size=meta["image_size"],
        segments=meta["segments"],
    )


@app.route("/download/<result_id>")
def download(result_id: str):
    path = _safe_path(RESULTS_FOLDER, result_id)
    if path is None or not path.exists():
        return jsonify(ok=False, error="Result file not found."), 404
    ext = path.suffix
    return send_file(path, as_attachment=True, download_name=f"ocr_result{ext}")


@app.errorhandler(413)
def too_large(_):
    return jsonify(ok=False, error="File too large. Maximum size is 20 MB."), 413


# ── Startup cleanup ───────────────────────────────────────────────────────────

_cleanup_old_files(UPLOAD_FOLDER)
_cleanup_old_files(RESULTS_FOLDER)

if __name__ == "__main__":
    # use_reloader=False: the reloader's file watcher restarts the worker
    # mid-request when /run lazily imports the heavy OCR stack (torch/cv2),
    # which kills the in-flight request and hangs the UI on "Processing…".
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
