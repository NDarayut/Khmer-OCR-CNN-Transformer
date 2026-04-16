from netra_ocr.ocr_engine import KhmerOCRPipeline

# 1. Initialize the pipeline (do this once to keep models in memory)
pipeline = KhmerOCRPipeline(engine="surya")

# 2. Call the function
result_text = pipeline.process_image(
    image_path="id_card.png",
    padding=4,
    beam_width=1,
    batch_size=16,
    output_path="id_card.txt",
    docx_flow=False,
)

print("OCR Result:")
print(result_text)