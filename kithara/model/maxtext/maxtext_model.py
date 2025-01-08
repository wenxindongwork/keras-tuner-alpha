from typing import Optional
from kithara.model.maxtext.ckpt_compatibility import (
    save_maxtext_model_in_hf_format,
    get_maxtext_model_name_from_hf_handle,
    load_hf_weights_into_maxtext_model,
)
from kithara.model.maxtext.conversion_utils import MaxTextConversionMixin
from kithara.model import Model, set_precision


class MaxTextModel(Model, MaxTextConversionMixin):
    """
    MaxTextModel is class that represents a MaxText model via the
    Kithara.Model interface. It is a thin wrapper around the underlying
    MaxText model instance.

    Methods
    -------
    from_random: Create a randomly initialized MaxText model with the
        given configuration.
    from_preset: Create a MaxText model initialized with weights from
        HuggingFace Hub.
    generate: Generate text based on the input tokens, with an option to
        stop at specific token IDs.
    save_in_hf_format: Save the MaxText model in HuggingFace format.
    """

    @classmethod
    def from_random(
        cls,
        model_name: str,
        seq_len: int = 2048,
        per_device_batch_size: int = 1,
        precision: str = "mixed_float16",
        scan_layers: bool = False,
        maxtext_config_args: Optional[dict] = None,
    ) -> "MaxTextModel":
        """Create a randomly initialized MaxText model with the given configuration.

        Args:
            model_name (str): Name of the MaxText model configuration to use.
            seq_len (int, optional): Maximum sequence length. Defaults to 2048.
            per_device_batch_size (int, optional): Batch size per device.
                Defaults to 1.
            precision (str, optional): Precision mode for computations.
                Defaults to "mixed_float16".
            scan_layers (bool, optional): Whether to use scan layers for memory efficiency.
                Defaults to False.
            maxtext_config_args (Optional[dict], optional): Additional configuration arguments.
                Defaults to None.

        Returns:
            MaxTextModel: A new instance of MaxTextModel with random initialization.
        """
        set_precision(precision)
        weight_dtype = cls._weight_dtype(precision)
        activation_dtype = cls._activation_dtype(precision)

        sharding_strategy, model = cls.initialize_random_maxtext_model(
            model_name,
            seq_len,
            per_device_batch_size,
            weight_dtype,
            activation_dtype,
            scan_layers,
            maxtext_config_args,
        )
        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
        )

    @classmethod
    def from_preset(
        cls,
        preset_handle: str,
        seq_len: int = 2048,
        per_device_batch_size: int = 1,
        precision: str = "mixed_float16",
        scan_layers: bool = False,
        maxtext_config_args: Optional[dict] = None,
    ) -> "MaxTextModel":
        """Create a MaxText model initialized with weights from HuggingFace Hub.

        Args:
            preset_handle (str): HuggingFace model identifier. This could be a
                HuggingFace Hub path (e.g "gs://google/gemma-2-2b), or
                a local HuggingFace checkpoint path (e.g. tmp/my_model/checkpoint), or
                a GCS HuggingFace checkpoint path (e.g. gs://bucket_name/my_model/checkpoint)
            seq_len (int): Maximum sequence length.
            per_device_batch_size (int): Batch size per device.
            precision (str, optional): Precision mode for computations.
                Defaults to "mixed_float16".
            scan_layers (bool, optional): Whether to use scan layers. Defaults to False.
            maxtext_config_args (Optional[dict], optional): Additional configuration arguments.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            MaxTextModel: A new instance of MaxTextModel initialized with pretrained weights.
        """

        set_precision(precision)
        weight_dtype = cls._weight_dtype(precision)
        activation_dtype = cls._activation_dtype(precision)

        model_name = get_maxtext_model_name_from_hf_handle(preset_handle)
        sharding_strategy, model = cls.initialize_random_maxtext_model(
            model_name,
            seq_len,
            per_device_batch_size,
            weight_dtype,
            activation_dtype,
            scan_layers,
            maxtext_config_args,
        )
        model = load_hf_weights_into_maxtext_model(preset_handle, model, scan_layers)

        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
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
            tokens_key="tokens",
            padding_mask_key="segment_ids",
        )

    def save_in_hf_format(self, output_dir: str, dtype: str = "auto"):
        """Save the model in HuggingFace format.

        Args:
            output_dir (str): Directory path where the model should be saved.
            dtype (str, optional): Data type for saved weights. Defaults to "auto".
        """
        save_maxtext_model_in_hf_format(self, output_dir, dtype=dtype)
