import argparse
import os
import sys
from .config import OCRConfig
from .utils import setup_logging, autodetect_config
from .cluster_tokenizer import ClusterTokenizer
from .predictor import OCRPredictor
from .model.model import KhmerOCR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# GLOBAL SETTINGS & STATE
# ==============================================================================
# The default recognition weight is *not* bundled with the package (it's
# ~77 MB) -- it's downloaded on first use from the HF model repo and cached
# locally by huggingface_hub (~/.cache/huggingface by default), so `pip
# install netra-ocr` stays small. Only the tiny vocab JSON is bundled, since
# it's needed immediately with no network round-trip.
DEFAULT_MODEL_REPO = "Darayut/khmer-text-recognition"
DEFAULT_MODEL_FILENAME = "khmerocr_cluster_ar.pth"
DEFAULT_VOCAB_PATH = os.path.join(CURRENT_DIR, "char2idx_cluster.json")

# Global variable to hold the model in memory (Singleton)
_PREDICTOR_INSTANCE = None


def _default_model_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=DEFAULT_MODEL_REPO, filename=DEFAULT_MODEL_FILENAME)


def _get_predictor(model_path=None, vocab_path=None):
    """
    Internal function to load the model only once.
    """
    global _PREDICTOR_INSTANCE

    if _PREDICTOR_INSTANCE is not None:
        return _PREDICTOR_INSTANCE

    # Use defaults if not provided. Resolving the default model path may
    # download it from the Hub, so this only runs once (guarded by the
    # singleton check above).
    model_path = model_path or _default_model_path()
    vocab_path = vocab_path or DEFAULT_VOCAB_PATH

    try:
        detected_cfg = autodetect_config(model_path)
        config = OCRConfig(**detected_cfg)
        tokenizer = ClusterTokenizer(vocab_path)

        _PREDICTOR_INSTANCE = OCRPredictor(
            model_path=model_path,
            tokenizer=tokenizer,
            config=config,
            model_class=KhmerOCR
        )
        return _PREDICTOR_INSTANCE

    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

# ==============================================================================
# PUBLIC API
# ==============================================================================

def recognize(image_input, beam_width: int = 3, model_path=None, vocab_path=None) -> str:
    """
    Recognizes text from an image path.
    
    Args:
        image_path (str): Path to the image file.
        beam_width (int): Beam search width (default 3).
        model_path (str): Optional override for model path.
        vocab_path (str): Optional override for vocab path.
    
    Returns:
        str: The predicted text.
    """
    predictor = _get_predictor(model_path, vocab_path)
    try:
        # Most predictors can handle PIL images. If yours requires a path, 
        # we'd need to modify predictor.py, but let's assume it handles objects.
        result_text = predictor.predict(image_input, beam_width=beam_width)
        return result_text
    except Exception as e:
        print(f"Prediction error: {e}")
        return ""

def recognize_batch(image_list: list, beam_width: int = 1, batch_size: int = 8, model_path=None, vocab_path=None) -> list:
    if not image_list:
        return []
    
    predictor = _get_predictor(model_path, vocab_path)
    try:
        # Pass the batch_size to the predictor
        return predictor.predict_batch(image_list, beam_width=beam_width, batch_size=batch_size)
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return [recognize(img, beam_width, model_path, vocab_path) for img in image_list]

# ===================
# CLI ENTRY POINT
# ===================
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Khmer OCR Inference Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth (default: auto-download from the HF model repo)")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH, help="Path to vocab json")
    parser.add_argument("--beam", type=int, default=3, help="Beam width (1 for greedy)")
    parser.add_argument("--output", type=str, help="Save result to text file")
    
    args = parser.parse_args()

    # Call the API function
    text = recognize(args.image, args.beam, args.model, args.vocab)
    
    print("\n" + "="*40)
    print(f"RESULT: {text}")
    print("="*40 + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

    """
    USAGE EXAMPLE:

        python recognize_text.py --image  "test_img_1.png"

        python recognize_text.py \
        --image "sample.png" \
        --model "weight/model.pth" \
        --vocab "char2idx.json" \
        --beam 5 \
        --output "results/sample_output.txt"
        
    ARGUMENTS:
        --image: Path to the input image (Required).
        --model: Path to .pth file (Default: auto-download from the HF model repo).
        --vocab: Path to .json vocab (Default: char2idx_cluster.json).
        --beam: Beam width. Set to 1 for Greedy Search (Default: 3).
        --output: (Optional) Text file to save the result.
    
    RUN VIA PYTHON:
        from recognize_text import recognize

        # Basic usage (uses defaults defined in the file)
        text = recognize("test_image_1.png")
        print(text)

        # Processing multiple images (Model stays loaded!)
        images = ["img1.png", "img2.png", "img3.png"]
        for img in images:
            print(f"{img}: {recognize(img)}")

        # Override settings if needed
        text_custom = recognize("test_image.png", beam_width=5, model_path="other_model.pth")
    """