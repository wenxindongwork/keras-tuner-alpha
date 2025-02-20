.. _model_api:

Model
=====

MaxTextModel
-----------

from_random
^^^^^^^^^^

.. py:classmethod:: MaxTextModel.from_random(model_name: str, seq_len: int = 2048, per_device_batch_size: int = 1, precision: str = "mixed_float16", scan_layers: bool = False, maxtext_config_args: Optional[dict] = None) -> "MaxTextModel"

   Create a randomly initialized MaxText model with the given configuration.

   :param model_name: Name of the MaxText model configuration to use. Supported: "default", "llama2-7b", "llama2-13b", "llama2-70b", "llama3-8b", "llama3-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "llama3.3-70b", "mistral-7b", "mixtral-8x7b", "mixtral-8x22b", "deepseek3-671b", "gemma-7b", "gemma-2b", "gemma2-2b", "gemma2-9b", "gemma2-27b", "gpt3-175b", "gpt3-22b", "gpt3-6b", "gpt3-52k"
   :param seq_len: Maximum sequence length (default: 2048)
   :param per_device_batch_size: Batch size per device (default: 1)
   :param precision: Precision mode for computations. Supported policies include "float32", "float16", "bfloat16", "mixed_float16", and "mixed_bfloat16". Mixed precision policies load model weight in float32 and casts activations to the specified dtype. (default: "mixed_float16")
   :param scan_layers: Whether to use scan layers for memory efficiency. Set to True for models <9B for performance gain. (default: False)
   :param maxtext_config_args: Additional MaxText configuration arguments (default: None)
   :return: A new instance of MaxTextModel with random initialization

   **Example Usage:**

   .. code-block:: python

      model = MaxTextModel.from_random(
         "gemma2-2b",
         seq_len=8192, # Seq len and batch size need to be specified up front
         per_device_batch_size=1, 
         scan_layers=True  # Set to True for models <9B for performance gain
      )

from_preset
^^^^^^^^^^

.. py:classmethod:: MaxTextModel.from_preset(preset_handle: str, seq_len: int = 2048, per_device_batch_size: int = 1, precision: str = "mixed_float16", scan_layers: bool = False, maxtext_config_args: Optional[dict] = None) -> "MaxTextModel"

   Create a MaxText model initialized with weights from HuggingFace Hub.

   :param preset_handle: HuggingFace model identifier for the supported model architectures. Can be:
                      - HuggingFace Hub path (e.g "gs://google/gemma-2-2b")
                      - Local HuggingFace checkpoint path (e.g. "tmp/my_model/checkpoint")
                      - GCS HuggingFace checkpoint path (e.g. "gs://bucket_name/my_model/checkpoint")
   :param seq_len: Maximum sequence length (default: 2048)
   :param per_device_batch_size: Batch size per device (default: 1)
   :param precision: Precision mode for computations. Supported policies include "float32", "float16", "bfloat16", "mixed_float16", and "mixed_bfloat16". Mixed precision policies load model weight in float32 and casts activations to the specified dtype. (default: "mixed_float16")
   :param scan_layers: Whether to use scan layers. Set to True for models <9B for performance gain. (default: False)
   :param maxtext_config_args: Additional configuration arguments (default: None)
   :return: A new instance of MaxTextModel initialized with pretrained weights

   **Example Usage:**

   .. code-block:: python

      model = MaxTextModel.from_preset(
         "hf://google/gemma-2-2b",  # HuggingFace model
         seq_len=8192, # Seq len and batch size need to be specified up front
         per_device_batch_size=1, 
         scan_layers=True  # Set to True for models <9B for performance gain
      )


save_in_hf_format
^^^^^^^^^^^^^^^

.. py:method:: save_in_hf_format(output_dir: str, dtype: str = "auto", parallel_threads: int = 8)

   Save the model in HuggingFace format, including:

   - Model configuration file (config.json)
   - Model weights file (model.safetensors for models smaller than DEFAULT_MAX_SHARD_SIZE, model-x-of-x.safetensors for larger models)
   - Safe tensors index file (model.safetensors.index.json)

   :param output_dir: Directory path where the model should be saved. Can be local or Google cloud storage path.
                   Will be created if it doesn't exist.
   :param dtype: Data type for saved weights. Defaults to "auto" which saves the model in its current precision type. (default: "auto")
   :param parallel_threads: Number of parallel threads to use for saving (default: 8).
                        Note: Local system must have at least parallel_threads * DEFAULT_MAX_SHARD_SIZE free disk space,
                        as each thread maintains a local cache of size DEFAULT_MAX_SHARD_SIZE

generate
^^^^^^^

.. py:method:: generate(inputs: Union[str | List[str] | Dict[str, np.ndarray]], max_length: int = 100, stop_token_ids: Union[str | List[int]] = "auto", strip_prompt: bool = False, tokenizer: Optional[AutoTokenizer] = None, tokenizer_handle: Optional[str] = None, return_decoded: bool = True, skip_special_tokens: bool = True, **kwargs) -> Union[List[str] | Dict[str, np.ndarray]]

   Generate text tokens using the model.

   :param inputs: A single string, a list of strings, or a dictionary with tokens as expected by the underlying model during the forward pass. If strings are provided, one of `tokenizer` and `tokenizer_handle` must be provided.
   :param max_length: Maximum total sequence length (prompt + generated tokens). If `tokenizer` and `tokenizer_handle` are `None`, `inputs` should be padded to the desired maximum length and this argument will be ignored. When `inputs` is string, this value must be provided. (default: 100)
   :param stop_token_ids: List of token IDs that stop generation. Defaults to "auto", which extracts the end token id from the tokenizer.
   :param strip_prompt: If True, returns only the generated tokens without the input prompt. If False, returns the full sequence including the prompt. (default: False)
   :param tokenizer: Optional AutoTokenizer instance.
   :param tokenizer_handle: Optional HuggingFace tokenizer identifier string. E.g. "google/gemma-2-2b".
   :param return_decoded: If True, returns the decoded text using the tokenizer, otherwise return the predicted tokens. (default: True). This option must be set to False if no tokenizer is provided.
   :param skip_special_tokens: Whether to remove special tokens from the decoded text. Only used when return_decoded is True. (default: True)
   :return: A list of strings if input is text, or a dictionary containing 'token_ids' (Generated token IDs [B, S]) and 'padding_mask' (Attention mask [B, S]) if return_decoded is False.

   **Example Usage:**

   .. code-block:: python

       # Return tokens
       prompt = "what is your name?"
       pred_tokens = model.generate(prompt, max_length=100, tokenizer_handle="hf://google/gemma-2-2b")
       print(pred_tokens)

       # Return text
       pred_text = model.generate(prompt, max_length=100, tokenizer_handle="hf://google/gemma-2-2b", 
                               return_decoded=True, strip_prompt=True)
       print(pred_text)

       # Use an initialized tokenizer
       from transformers import AutoTokenizer
       tokenizer = AutoTokenizer.from_pretrained("hf://google/gemma-2-2b")  
       pred_text = model.generate(prompt, max_length=100, tokenizer=tokenizer,
                               return_decoded=True, strip_prompt=True)


KerasHubModel
------------

from_preset
^^^^^^^^^^

.. py:classmethod:: KerasHubModel.from_preset(preset_handle: str, lora_rank: Optional[int] = None, precision: str = "mixed_float16", sharding_strategy: Optional[ShardingStrategy] = None, **kwargs) -> "KerasHubModel"

   Create a KerasHub model initialized with weights from various sources, with optional LoRA adaptation.

   :param preset_handle: Model identifier that can be:
                      - A built-in KerasHub preset identifier (e.g., "bert_base_en")
                      - A Kaggle Models handle (e.g., "kaggle://user/bert/keras/bert_base_en")
                      - A Hugging Face handle (e.g., "hf://user/bert_base_en")
                      - A local directory path (e.g., "./bert_base_en")
   :param lora_rank: Rank for LoRA adaptation. If None, LoRA is disabled. When enabled, LoRA is applied to the q_proj and v_proj layers. (default: None)
   :param precision: Precision mode for computations. Supported policies include "float32", "float16", "bfloat16", "mixed_float16", and "mixed_bfloat16". Mixed precision policies load model weights in float32 and cast activations to the specified dtype. (default: "mixed_float16")
   :param sharding_strategy: Strategy for distributing model parameters, optimizer states, and data tensors. If None, tensors will be sharded using FSDP. Use kithara.ShardingStrategy to configure custom sharding. (default: None)
   :return: A new instance of KerasHubModel initialized with the specified configuration

   **Example Usage:**

   .. code-block:: python

       # Initialize a model with LoRA adaptation
       model = KerasHubModel.from_preset(
           "hf://google/gemma-2-2b",
           lora_rank=4
       )

save_in_hf_format
^^^^^^^^^^^^^^^

.. py:method:: save_in_hf_format(output_dir: str, dtype: str = "auto", only_save_adapters: bool = False, save_adapters_separately: bool = False, parallel_threads: int = 8)

   Save the model in HuggingFace format, including configuration and weights files.

   :param output_dir: Directory path where the model should be saved. Can be local or Google cloud storage path.
                   Will be created if it doesn't exist.
   :param dtype: Data type for saved weights. Defaults to "auto" which saves the model in its current precision type.
   :param only_save_adapters: If True, only adapter weights will be saved. If False, both base model weights and adapter weights will be saved. (default: False)
   :param save_adapters_separately: If False, adapter weights will be merged with base model. If True, adapter weights will be saved separately in HuggingFace's peft format. (default: False)
   :param parallel_threads: Number of parallel threads to use for saving (default: 8).
                        Note: Local system must have at least parallel_threads * DEFAULT_MAX_SHARD_SIZE free disk space,
                        as each thread maintains a local cache of size DEFAULT_MAX_SHARD_SIZE

   **Example Usage:**

   .. code-block:: python

       # Save full model
       model.save_in_hf_format("./output_dir")

       # Save only LoRA adapters
       model.save_in_hf_format(
           "./adapter_weights",
           only_save_adapters=True,
       )

generate
^^^^^^^

.. py:method:: generate(inputs: Union[str | List[str] | Dict[str, np.ndarray]], max_length: int = 100, stop_token_ids: Union[str | List[int]] = "auto", strip_prompt: bool = False, tokenizer: Optional[AutoTokenizer] = None, tokenizer_handle: Optional[str] = None, return_decoded: bool = True, skip_special_tokens: bool = True, **kwargs) -> Union[List[str] | Dict[str, np.ndarray]]

   Generate text tokens using the model.

   :param inputs: A single string, a list of strings, or a dictionary with tokens as expected by the underlying model during the forward pass. If strings are provided, one of `tokenizer` and `tokenizer_handle` must be provided.
   :param max_length: Maximum total sequence length (prompt + generated tokens). If `tokenizer` and `tokenizer_handle` are `None`, `inputs` should be padded to the desired maximum length and this argument will be ignored. When `inputs` is string, this value must be provided. (default: 100)
   :param stop_token_ids: List of token IDs that stop generation. Defaults to "auto", which extracts the end token id from the tokenizer.
   :param strip_prompt: If True, returns only the generated tokens without the input prompt. If False, returns the full sequence including the prompt. (default: False)
   :param tokenizer: Optional AutoTokenizer instance.
   :param tokenizer_handle: Optional HuggingFace tokenizer identifier string. E.g. "google/gemma-2-2b".
   :param return_decoded: If True, returns the decoded text using the tokenizer, otherwise return the predicted tokens. (default: True). This option must be set to False if no tokenizer is provided.
   :param skip_special_tokens: Whether to remove special tokens from the decoded text. Only used when return_decoded is True. (default: True)
   :return: A list of strings if input is text, or a dictionary containing 'token_ids' (Generated token IDs [B, S]) and 'padding_mask' (Attention mask [B, S]) if return_decoded is False.

   **Example Usage:**

   .. code-block:: python

       # Return tokens
       prompt = "what is your name?"
       pred_tokens = model.generate(prompt, max_length=100, tokenizer_handle="hf://google/gemma-2-2b")
       print(pred_tokens)

       # Return text
       pred_text = model.generate(prompt, max_length=100, tokenizer_handle="hf://google/gemma-2-2b", 
                               return_decoded=True, strip_prompt=True)
       print(pred_text)

       # Use an initialized tokenizer
       from transformers import AutoTokenizer
       tokenizer = AutoTokenizer.from_pretrained("hf://google/gemma-2-2b")  
       pred_text = model.generate(prompt, max_length=100, tokenizer=tokenizer,
                               return_decoded=True, strip_prompt=True)
