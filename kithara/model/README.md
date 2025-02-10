# Kithara Model Support Guide

## Overview

Kithara is designed as a model-agnostic framework that focuses on model tuning rather than implementation. It provides integration with existing model implementations through supported providers:

- MaxText
- KerasHub

Additional model implementation providers may be added in the future, community contributions are always welcomed. 

## Supported Models

The complete list of supported models is maintained in `supported_models.py`. For the most up-to-date information, please refer to this file.

## Adding Support for New Models

This guide focuses on adding support for new KerasHub model architectures. The process involves several key steps:

### 1. Configure Model Sharding

1. Navigate to `distributed/sharding/models/`
2. Create or modify the FSDP (Fully Sharded Data Parallel) model sharding strategy for your model
3. Update the `PredefinedShardingStrategy` class to include support for your model

### 2. Model Implementation

First, create a KerasHub model instance to examine its structure:

```python
model = KerasHubModel.from_preset(
    your_model_handle
)

# Examine model weights and shapes
for weight in model.weights:
    print(f"Path: {weight.path}, Shape: {weight.value.shape}")
```

### 3. HuggingFace Compatibility

#### 3.1 Analyze HuggingFace Model Structure

Create and examine the equivalent HuggingFace model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(preset_handle)

# Examine model state dictionary
for key, value in model.state_dict().items():
    print(f"Key: {key}, Shape: {value.shape}")
```

#### 3.2 Update Configuration Files

1. Add HuggingFace model configuration:
   - File: `hf_compatibility/model_configs.py`
   - Add the specific configuration for your model

2. Define weight shapes:
   - File: `hf_compatibility/shape_mapping.py`
   - Document the shapes of all weights in the HuggingFace model

3. Create weight mappings:
   - File: `model/kerashub/ckpt_compatibility/param_mapping.py`
   - Define the mapping between HuggingFace and Kithara weights
   - Implement weight translation hook functions

### 4. Testing and Validation

1. Create unit tests in `tests/model/models/kerashub/ckpt_compatibility/`
2. Follow the pattern established in `gemma2-2b.py`
3. Verify numerical correctness of checkpoint translations

### Debugging Tips

When working on weight translations:

1. Fork the transformers library for debugging
2. Add logging statements to compare intermediate tensor outputs
3. Verify tensor values at each later

## Best Practices

1. Support all sizes of the supported model family
2. Document all weight mappings thoroughly and include examples of expected tensor shapes
