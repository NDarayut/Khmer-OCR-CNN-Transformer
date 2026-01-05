from typing import Optional

from transformers import PreTrainedModel
from transformers.utils import is_flash_attn_2_available


class SuryaPreTrainedModel(PreTrainedModel):
    # No-op if we pass attention, so we can set attention however we want in the config
    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], **kwargs
    ):
        if attn_implementation is None:
            try:
                self._sdpa_can_dispatch(True)
                attn_implementation = "sdpa"
            except (ValueError, ImportError):
                attn_implementation = "eager"

            if self._supports_flash_attn and is_flash_attn_2_available():
                attn_implementation = "flash_attention_2"

        return attn_implementation
