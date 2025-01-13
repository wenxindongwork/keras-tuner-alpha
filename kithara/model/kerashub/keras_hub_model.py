from typing import Optional
from keras_nlp.models import CausalLM
from kithara.distributed.sharding import ShardingStrategy
from kithara.model.model import Model, set_precision, set_global_sharding_strategy
from kithara.model.kerashub.ckpt_compatibility.to_huggingface import (
    save_kerashub_model_in_hf_format,
)
from kithara.model.hf_compatibility import get_model_name_from_preset_handle


class KerasHubModel(Model):
    """
    A Kithara model wrapper for KerasHub models.

    Attributes:
        model_handle (str): Model identifier, e.g., "hf://google/gemma-2-2b".
        lora_rank (Optional[int]): Rank for LoRA adaptation (disabled if None), applied to q_proj and v_proj
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
        """Load a KerasHub model, optionally apply LoRA, and configure precision and sharding.

        Args:
            model_handle (str): Identifier for the model preset. This can be:
                - A built-in KerasHub preset identifier (e.g., `"bert_base_en"`).
                - A Kaggle Models handle (e.g., `"kaggle://user/bert/keras/bert_base_en"`).
                - A Hugging Face handle (e.g., `"hf://user/bert_base_en"`).
                - A local directory path (e.g., `"./bert_base_en"`).
            lora_rank (Optional[int]): Rank for LoRA adaptation. If None, LoRA is disabled.
                Defaults to None. When enabled, LoRA is applied to the `q_proj` and `v_proj` layers.
            precision (str): Precision policy for the model. Defaults to "mixed_float16".
                Supported options include: "float32", "float16", "bfloat16", "mixed_float16",
                and "mixed_bfloat16". Mixed precision policies load weights in float32 and cast
                activations to the specified dtype.
            sharding_strategy (Optional[ShardingStrategy]): Strategy for distributing model parameters,
                optimizer states, and data tensors. If None, model will be replicated across all devices.
                Defaults to None. You can use `kithara.PredefinedShardingStrategy` to easily
                configure common sharding strategies.

        Returns:
            KerasHubModel: An instance of the `KerasHubModel` class.

        Example:
            ```
            model = KerasHubModel.from_preset(
                "hf://google/gemma-2-2b",
                lora_rank=4,
                sharding_strategy=kithara.PredefinedShardingStrategy("fsdp", "gemma")
            )
            ```
        """
        set_precision(precision)
        set_global_sharding_strategy(sharding_strategy)
        model_name = get_model_name_from_preset_handle(model_handle)

        model = CausalLM.from_preset(model_handle, preprocessor=None, **kwargs)
        if lora_rank:
            model.backbone.enable_lora(rank=lora_rank)

        return cls(
            model,
            model_name=model_name,
            sharding_strategy=sharding_strategy,
            precision=precision,
            lora_rank=lora_rank,
        )

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids=None,
        strip_prompt=False,
    ):
        return self._generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
            tokens_key="token_ids",
            padding_mask_key="padding_mask",
        )

    def save_in_hf_format(
        self,
        output_dir: str,
        dtype: str = "auto",
        only_save_adapters=False,
        save_adapters_separately=False,
        parallel_threads=8,
    ):
        """Save the model in HuggingFace format, including the model configuration file (`config.json`),
            the model weights file (`model.safetensors` for models smaller than
            `DEFAULT_MAX_SHARD_SIZE` and `model-x-of-x.safetensors` for larger models),
            and the safe tensors index file (`model.safetensors.index.json`).

        Args:
            output_dir (str): Directory path where the model should be saved.
                Directory could be local or a Google cloud storage path, and will be created if
                it doesn't exist.
            dtype (str, optional): Data type for saved weights. Defaults to "auto".
            only_save_adapters (bool): If set to True, only adapter weights will be saved. If
                set to False, both base model weights and adapter weights will be saved. Default
                to False.
            save_adapters_separately (bool): If set to False, adapter weights will be merged with base model.
                If set to True, adapter weights will be saved separately in HuggingFace's peft format.
                Default to False.
            parallel_threads (int, optional): Number of parallel threads to use for saving.
                Defaults to 8. Make sure the local system has at least
                `parallel_threads * DEFAULT_MAX_SHARD_SIZE` free disk space,
                as each thread will maintain a local cache of size `DEFAULT_MAX_SHARD_SIZE`.
        """
        save_kerashub_model_in_hf_format(
            self,
            output_dir,
            dtype=dtype,
            only_save_adapters=only_save_adapters,
            save_adapters_separately=save_adapters_separately,
            parallel_threads=parallel_threads,
        )
