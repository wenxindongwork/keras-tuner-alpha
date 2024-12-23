from typing import Optional
from keras_nlp.models import CausalLM
from kithara.distributed.sharding import ShardingStrategy
from kithara.model.model import Model, set_precision, set_global_sharding_strategy


class KerasHubModel(Model):
    """
    A Kithara model wrapper for KerasHub models.

    Attributes:
        model_handle (str): Model identifier, e.g., "hf://google/gemma-2-2b".
        lora_rank (Optional[int]): Rank for LoRA adaptation (disabled if None).
        sharding_strategy(kithara.ShardingStrategy): Strategy used for distributing model, optimizer, 
            and data tensors. E.g. `kithara.PredefinedShardingStrategy("fsdp", "gemma")`.
            Default is "mixed_bfloat16". Supported policies include "float32", "float16", "bfloat16", 
            "mixed_float16", and "mixed_bfloat16". Mixed precision policies load model weight in float32 
            and casts activations to the specified dtype.
    
    Example Usage:
        model = KerasHubModel.from_preset("hf://google/gemma-2-2b", lora_rank=4, 
                sharding_strategy=kithara.PredefinedShardingStrategy("fsdp", "gemma"))
    """

    @classmethod
    def from_preset(
        cls,
        model_handle: str,
        lora_rank: Optional[int] = None,
        precision: str = "mixed_float16",
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
            sharding_strategy=sharding_strategy,
            precision=precision,
        )
    
    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids=None,
        strip_prompt=False,
    ):
        return self._generate(inputs, 
                             max_length=max_length, 
                             stop_token_ids=stop_token_ids, 
                             strip_prompt=strip_prompt, 
                             tokens_key = "token_ids",
                             padding_mask_key = "padding_mask")

