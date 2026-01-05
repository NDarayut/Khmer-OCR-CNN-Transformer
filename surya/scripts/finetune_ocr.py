from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from datasets import load_dataset
import numpy as np
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

from surya.common.surya import SuryaModel
from surya.common.surya.processor import SuryaOCRProcessor
from surya.foundation import FoundationPredictor
from surya.common.surya.processor.schema import ImageInput, TextInput
from surya.common.surya.schema import TaskNames
from surya.common.util import get_top_scripts, SCRIPT_TOKEN_MAPPING

# Do not change these defaults
OCR_TASK_NAME = TaskNames.ocr_with_boxes
OCR_MAX_IMAGE_SIZE = (1024, 512)

# Simple wrapper for huggingface dataset
class SuryaOCRDataset(torch.utils.data.Dataset):
    def __init__(self, processor: SuryaOCRProcessor, data_args: SuryaOCRDataArguments):
        super().__init__()
        self.hf_dataset = load_dataset(data_args.dataset_name, num_proc=data_args.num_loading_proc, split="train")
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def get_script_text(self, text: str) -> str:
        scripts = get_top_scripts(text)
        script_text = "".join(SCRIPT_TOKEN_MAPPING[script] for script in scripts)
        return script_text

    def __getitem__(self, index):
        try:
            data = self.hf_dataset[index]
            image = data["image"]
            image = image.convert("RGB")
            image = np.asarray(image, dtype=np.float32)
            image = self.processor.scale_to_fit(image, max_size=OCR_MAX_IMAGE_SIZE)

            # Add in script information
            gt_text = data["text"]
            gt_text = self.get_script_text(gt_text) + gt_text

            return_dict = {
                "task": TaskNames.ocr_with_boxes,
                "inputs": [
                    ImageInput(type="image", image=image, rotated=False),
                    # This empty TextInput **must be included** to match the original format
                    TextInput(type="text", text=""),
                    TextInput(type="text",text=gt_text),
                ],
            }
            return return_dict
        except:
            import traceback; traceback.print_exc()
            return self.__getitem__((index + 1) % self.__len__())

class SuryaOCRDataCollator:
    def __init__(self, processor: SuryaOCRProcessor, data_args: SuryaOCRDataArguments):
        self.processor = processor
        self.max_sequence_length = data_args.max_sequence_length

    def __call__(self, inputs):
        # Use right padding for training. Defaults to left for inference
        processed_batch = self.processor(inputs, padding_side="right")
        
        if self.max_sequence_length is not None:
            processed_batch["input_ids"] = processed_batch["input_ids"][:, :self.max_sequence_length]
            processed_batch["attention_mask"] = processed_batch["attention_mask"][:, :self.max_sequence_length]
            processed_batch["position_ids"] = processed_batch["position_ids"][:, :self.max_sequence_length]

        lm_labels = processed_batch["input_ids"].clone()
        skip_label_mask = (
            (lm_labels == self.processor.pad_token_id )
            | (lm_labels == self.processor.bos_token_id[TaskNames.ocr_with_boxes])
            | (lm_labels == self.processor.eoi_token_id)
            | (lm_labels == self.processor.image_token_id)
        )
        lm_labels[skip_label_mask] = -100
        processed_batch["labels"] = lm_labels

        return processed_batch

def load_model_and_processor(checkpoint_path: Optional[str] = None) -> Tuple[SuryaModel, SuryaOCRProcessor]:
    foundation_predictor = FoundationPredictor(checkpoint=checkpoint_path)
    return foundation_predictor.model, foundation_predictor.processor

@dataclass
class SuryaOCRModelArguments:
    pretrained_checkpoint_path: Optional[str] = field(default=None)

@dataclass
class SuryaOCRDataArguments:
    dataset_name: str = field(default="datalab-to/ocr_finetune_example")
    num_loading_proc: int = field(default=16)
    max_sequence_length: Optional[int] = field(default=None)

@dataclass
class SuryaOCRTrainingArguments(TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    
def main():
    parser = HfArgumentParser((SuryaOCRModelArguments, SuryaOCRDataArguments, SuryaOCRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = load_model_and_processor(model_args.pretrained_checkpoint_path)
    dataset = SuryaOCRDataset(processor, data_args)
    collator = SuryaOCRDataCollator(processor, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    trainer.train()

if __name__ == "__main__":
    main()