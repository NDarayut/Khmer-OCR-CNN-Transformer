<div align="center">
  <img src="assets/netra-logo-transparent-new.png" width="60%" alt="Netra Lab" />
</div>

<hr>

<p align="center">
 <a href="https://github.com/netra-ai-lab/Khmer-OCR-CNN-Transformer"><b>GitHub</b></a> |
  <a href="https://huggingface.co/Darayut/khmer-text-recognition"><b>Model Download</b></a> |
    <a href="https://huggingface.co/collections/Darayut/khmer-text-synthetic"><b>Dataset Download</b></a> |
    <a href="https://huggingface.co/spaces/Darayut/Khmer-Text-Recognition"><b>Inference Space</b></a> |
</p>


<h2>
<p align="center">
  <a href="">A Squeeze-and-Excitation Transformer Network for Khmer Optical Character Recognition</a>
</p>
</h2>

<p align="center">
<img src="assets/benchmark.png" style="width: 1000px" align=center>
</p>

<p align="center">
<a href="">Character Error Rate (CER %) on KHOB, Legal Documents, and Printed Word Benchmark</a>       
</p>

## Introduction

This repository contains the implementation, datasets, and evaluation results for the **Squeeze-and-Excitation Transformer Network**, a high-performance Khmer Text Recognition model that utilizes a hybrid architecture combining **Squeeze-and-Excitation** blocks for feature extraction and **BiLSTM** smoothing for context smoothing, specifically designed to handle the complexity and length of Khmer script.

## Overview

Khmer script presents unique challenges for OCR due to its large character set, complex sub-consonant stacking, and variable text line lengths. This project proposes an enhanced pipeline that:
1.  **Chunks** long text lines into manageable overlapping segments.
2.  **Extracts Features** using a **Squeeze-and-Excitation Network** (SE-VGG) that preserves horizontal spatial information.
3.  **Encodes** local spatial features using a Transformer Encoder.
4.  **Merges** the encoded chunks into a unified sequence.
5.  **Smooths Context** using a **BiLSTM** layer to resolve boundary discontinuities between chunks.
6.  **Decodes** the final sequence using a Transformer Decoder.

## Datasets

The model was trained entirely on synthetic data and evaluated on real-world datasets.
### Training Data (Synthetic)
We generated **200,000 synthetic images** to ensure robustness against font variations and background noise.

| Dataset Type | Count | Generator / Source | Augmentations |
| :--- | :--- | :--- | :--- |
| **Document Text** | 100,000 | Pillow + Khmer Corpus | Erosion, noise, thinning/thickening, perspective distortion. |
| **Scene Text** | 100,000 | SynthTIGER + Stanford BG | Rotation, blur, noise, realistic backgrounds. |

### Evaluation Data (Real-World + Synthetic)
| Dataset | Type | Size | Description |
| :--- | :--- | :--- | :--- |
| **KHOB** | Real | 325 | Standard benchmark, clean backgrounds but compression artifacts. |
| **Legal Documents** | Real | 227 | High variation in degradation, illumination, and distortion. |
| **Printed Words** | Synthetic | 1,000 | Short, isolated words in 10 different fonts. |

![Dataset Overview](/assets/dataset-overview.png)
---

## Methodology & Architecture

### 1. Preprocessing: Chunking & Merging
To handle variable-length text lines without aggressive resizing, we employ a "Chunk-and-Merge" strategy:
*   **Resize:** Input images are resized to a fixed height of 48 pixels while maintaining aspect ratio.
*   **Chunking:** The image is split into overlapping chunks (Size: 48x100 px, Overlap: 16 px).
*   **Independent Encoding:** Each chunk is processed independently by the Squeeze-and-Excitation Network and Transformer Encoder to allow for parallel batch processing.

### 2. Model Architecture: Squeeze-and-Excitation Transformer Network
Our proposed architecture integrates sequence-aware attention and recurrent smoothing to overcome the limitations of standard chunk-based OCR. The model consists of six key modules:

![Model Architecture](/assets/ocr-architecture.png)

1.  **Squeeze-and-Excitation Network (SE-VGG):**
    *   A modified VGG backbone with **1D Squeeze-and-Excitation** blocks after convolutional layer **3**, **4**, and **5**.
    *   Unlike standard SE, these blocks use **vertical pooling** to refine feature channels while strictly preserving the horizontal width (sequence information).

        ![SE Module](</assets/Sequence Attention CNN.png>)


2.  **Patch Module:**
    *   Projects spatial features into a condensed **384-dimensional** embedding space.
    *   Adds local positional encodings to preserve spatial order within chunks.

3.  **Transformer Encoder:**
    *   Captures contextual relationships among visual tokens within each independent chunk.

4.  **Merging Module:**
    *   Concatenates the encoded features from all chunks into a single unified sequence.
    *   Adds **Global Positional Embeddings** to define the absolute position of tokens across the entire text line.

5.  **BiLSTM Context Smoother:**
    *   A Bidirectional LSTM layer that processes the merged sequence.
    *   **Purpose:** Bridges the "context gap" between independent chunks by smoothing boundary discontinuities, ensuring a seamless flow of information across the text line.

        ![Context Smoothing Module](/assets/BiLSTM-Module.png)

6.  **Transformer Decoder:**
    *   Generates the final Khmer character sequence using the globally smoothed context.

---

## Training Configuration

*   **Epochs:** 100
*   **Optimizer:** Adam
*   **Loss Function:** Cross-Entropy Loss
*   **Learning Rate Schedule:** Staged Cyclic
    *   *Epoch 0-15:* Fixed 1e-4 (Rapid convergence)
    *   *Epoch 16-30:* Cyclic 1e-4 to 1e-5 (Stability)
    *   *Epoch 31-100:* Cyclic 1e-5 to 1e-6 (Fine-tuning)
*   **Sampling:** 50,000 images randomly sampled/augmented per epoch.

---

## Quantitative Analysis

We benchmarked our **proposed model** against VGG-Transformer, ResNet-Transformer, and Tesseract-OCR.

**Character Error Rate (CER %)** - *Lower is better*

TABLE 1: Character Error Rate (CER in %) results on the KHOB, Legal Documents, and Printed Word

| Model | KHOB | Legal Documents | Printed Word |
| :--- | :--- | :--- | :--- |
| Tesseract-OCR | 6.24 | 24.30 | 8.02 |
| VGG-Transformer | 2.27 | 10.27 | 3.61 |
| ResNet-Transformer | 2.98 | 11.57 | 2.80 |
| **Proposed Model** | $\textcolor{yellow}{1.87}$ | $\textcolor{yellow}{9.13}$ | $\textcolor{yellow}{2.46}$ |

---

## Qualitative Analysis

TABLE 2: Failure cases on KHOB, Legal Document, and Printed Word dataset
![failure cases](/assets/failure_cases.png)

TABLE 3: Example of proposed, and baseline model compared with the ground truth. Errors in the predictions are highlighted in red
![success cases](/assets/sucess_case.png)

**Key Findings:**
*   **The Proposed Model** achieves the highest accuracy on long, continuous text lines (KHOB), demonstrating that the **BiLSTM Context Smoother** effectively resolves the chunk boundary discontinuities that limit standard Transformer baselines.
*   On degraded and complex legal documents, **the proposed model** demonstrates superior robustness, significantly outperforming all baselines. This attributes to the **Squeeze-and-Excitation blocks**, which filter background noise while preserving character-specific features.
*   **The Proposed Model** still retains a slight advantage on short, isolated words even where global context is less critical, outperforming both ResNet and VGG Transformer baseline.


## Setup

### Create virtual environment
```bash
# Windows
python -m venv myenv
.\myenv\Scripts\activate

# Mac/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### Installation
```bash
# From PyPI
pip install netra-ocr

# Or install the latest from GitHub
pip install -v git+https://github.com/netra-ai-lab/Khmer-OCR-CNN-Transformer.git@master
```

The default **YOLO** and **legacy** detectors and the recognition model work out of the
box — the default SE-Transformer weights (`khmerocr_epoch570.pth`) and the YOLO detector
weights are bundled with the package (~88 MB). Other trained checkpoints listed in the
[model page](https://huggingface.co/Darayut/khmer-text-recognition) are not bundled; pass
them explicitly via `--model` / `model_path` if you want to use them.

#### Optional extras
```bash
# Tesseract detector backend (also requires the system Tesseract binary)
pip install "netra-ocr[tesseract]"

# Flask browser UI (app.py)
pip install "netra-ocr[web]"
```

---
## Inference Usage
This pipeline performs Khmer OCR — it detects text lines (and optionally logos) in a document image and extracts the recognized text into your chosen output format.

### Supported Output Formats
The output format is selected automatically from the file extension:
- **`.txt` / `.md`** — plain UTF-8 text, one line per detected text line.
- **`.docx`** — Word document: text lines as paragraphs, detected logos embedded as inline images.
- **`.json`** — structured metadata including image size and per-line text + bounding boxes, suitable for reconstructing the document layout later.

### Detectors
| Detector | Description |
| :--- | :--- |
| `yolo` (default) | YOLOv26s trained on Khmer documents. Detects **class 0** (text lines) and **class 1** (logos). Logos are cropped and embedded in `.docx` output; other formats receive text only. Text boxes are refined after detection to horizontally cover the full text line (content-aware, on by default). |
| `tesseract` | Tesseract + graph clustering. No external model required. |
| `legacy` | Classic CV detector using MSER, gradient analysis, and multi-channel binarization. No GPU or Tesseract installation required. Accepts optional `pad` parameter. |

### Recognition Post-Processing
Raw decoder output is cleaned of common gibberish (control/replacement characters and runaway repeated characters, clusters, or tokens from decoder loops). Machine-readable zone (MRZ) lines on passports/ID cards are auto-detected and exempted, so legitimate repeated `<` filler (e.g. `IDKHM1011052875<<<<<<<<`) is preserved intact.

---

## Local-Inference

### 1. Command Line Interface (CLI)
```bash
netra_ocr --image path/to/your/image.jpg --output result.txt
```

#### More examples
```bash
# Classic CV detector — no GPU or Tesseract required
netra_ocr --image scan.jpg --output result.txt --detector legacy

# Legacy with custom padding
netra_ocr --image scan.jpg --output result.txt --detector legacy --pad 4

# Save as Word document (YOLO: logos are embedded as images)
netra_ocr --image scan.jpg --output result.docx --detector yolo

# Save structured JSON (includes bbox per line, text only)
netra_ocr --image scan.jpg --output result.json --detector yolo

# Tune YOLO confidence threshold (default 0.25)
netra_ocr --image scan.jpg --output result.txt --detector yolo --conf 0.4

# High-accuracy mode with Tesseract detector
netra_ocr --image scan.jpg --output result.txt --detector tesseract --beam 5 --batch_size 16

# Debug mode — saves per-line .txt and logo .png files to a debug_ folder
netra_ocr --image scan.jpg --output result.txt --detector yolo --debug
```

### 2. Python API
Instantiate `KhmerOCRPipeline` once to keep models in memory for repeated calls.

```python
from netra_ocr.ocr_engine import KhmerOCRPipeline

# YOLO detector with custom confidence threshold
pipeline = KhmerOCRPipeline(detector="yolo", conf=0.4)

# Process an image — returns recognized text; also writes the output file
result_text = pipeline.process_image(
    image_path="document.png",
    output_path="document.docx",   # Extension determines format
    beam_width=1,
    batch_size=8,
    save_debug=False,
)

print(result_text)
```

### 3. Web App (Browser UI)
A Flask web interface ([`app.py`](app.py)) for uploading an image, picking a detector/output format, and viewing the recognized text with bounding-box overlays in the browser.

```bash
pip install flask
python app.py
# open http://localhost:5000
```

Upload a JPG/PNG/TIFF/BMP (max 20 MB), choose a detector (`tesseract`, `yolo`, or `legacy`) and output format (`.txt`, `.md`, `.json`, `.docx`), then run OCR and download the result. Pipelines are cached in memory per detector configuration for fast repeat runs.

---

### CLI & API Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image_path` | `str` | **Required** | Path to the input image file. |
| `detector` | `str` | `yolo` | Text detector: `yolo`, `tesseract`, or `legacy`. |
| `conf` | `float` | `0.25` | YOLO confidence threshold. Only applies when `detector="yolo"`. |
| `pad` | `int` | `None` (auto) | Pixels added around each detected box. Applies to `yolo` and `legacy` detectors. |
| `output_path` | `str` | `None` | Destination file. Extension selects format: `.txt`, `.md`, `.json`, `.docx`. |
| `beam_width` | `int` | `1` | `1` = greedy search (fast). Higher values improve accuracy at the cost of speed. |
| `batch_size` | `int` | `8` | Number of text lines processed per recognition batch. |
| `save_debug` | `bool` | `False` | Saves per-segment debug files (`.txt` for text, `.png` for logos) into a `debug_<name>/` folder. |


## Huggingface-Inference
1. Setup
```bash
pip install torch torchvision transformers pillow huggingface_hub

# Setup the inference script
wget https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/configuration_khmerocr.py

wget https://huggingface.co/Darayut/khmer-text-recognition/resolve/main/inference.py

```
2. Run via CLI
```bash
python inference.py --image "path/to/image.png" --method beam --beam_width 3

```

3. Run via Python
```python
from inference import KhmerOCR

# Load Model (Downloads automatically)
ocr = KhmerOCR()

# Predict
text = ocr.predict("test_image.jpg", method="beam", beam_width=3)
print(text)

```

---
## References

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**  
   *Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.*  
   ICLR 2021.  
   [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models**  
   *Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei.*  
   AAAI 2023.  
   [arXiv:2109.10282](https://arxiv.org/abs/2109.10282)

3. **Toward a Low-Resource Non-Latin-Complete Baseline: An Exploration of Khmer Optical Character Recognition**  
   *R. Buoy, M. Iwamura, S. Srun and K. Kise.*  
   IEEE Access, vol. 11, pp. 128044-128060, 2023.  
   [DOI: 10.1109/ACCESS.2023.3332361](https://doi.org/10.1109/ACCESS.2023.3332361)

4. **Balraj98.** (2018). *Stanford background dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/balraj98/stanford-background-dataset

5. **EKYC Solutions.** (n.d.). *Khmer OCR benchmark dataset (KHOB)* [Data set]. GitHub. https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset

6. **Em, H., Valy, D., Gosselin, B., & Kong, P.** (2024). *Khmer text recognition dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/emhengly/khmer-text-recognition-dataset

7. **Squeeze-and-Excitation Networks**  
   *Jie Hu, Li Shen, and Gang Sun.*  
   CVPR 2018.  
   [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)

8. **Bidirectional Recurrent Neural Networks**  
   *Mike Schuster and Kuldip K. Paliwal.*  
   IEEE Transactions on Signal Processing, 1997.  
   [DOI: 10.1109/78.650093](https://doi.org/10.1109/78.650093)