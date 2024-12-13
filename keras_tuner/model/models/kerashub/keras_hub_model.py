from typing import Optional
from keras_nlp.models import CausalLM
from keras_tuner.model.sharding import ShardingStrategy
from keras_tuner.model import Model, set_precision, set_global_sharding_strategy

class KerasHubModel(Model):
    """
    A Kithara model wrapper for KerasHub models, providing a Keras-compatible interface.

    Attributes:
        model_handle (str): Model identifier, e.g., "hf://google/gemma-2-2b".
        lora_rank (Optional[int]): Rank for LoRA adaptation (disabled if None).
    """

    @classmethod
    def from_preset(
        cls,
        model_handle: str,
        lora_rank: Optional[int] = None,
        precision: Optional[str] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        **kwargs,
    ) -> "KerasHubModel":
        """Load a Keras model from a preset and apply LoRA if specified."""
        set_precision(precision)
        set_global_sharding_strategy(sharding_strategy)

        model = CausalLM.from_preset(model_handle, preprocessor=None, **kwargs)
        if lora_rank:
            model.backbone.enable_lora(rank=lora_rank)

        return cls(
            model,
            sharding_strategy,
            precision=precision,
        )
