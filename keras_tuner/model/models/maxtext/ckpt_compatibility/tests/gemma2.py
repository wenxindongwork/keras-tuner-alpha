"""Unit test for model conversion correctness

Run test with: `pytest gemma2.py -v`
"""
import os
import pytest
import numpy as np
import torch
from keras_tuner.model import MaxTextModel
from transformers import AutoModelForCausalLM
from typing import List, Tuple, Dict

# Set backend
os.environ["KERAS_BACKEND"] = "jax"

# Constants
TMP_DIR = '/dev/shm/temp/hf/checkpoint/'

# Model configurations
MODEL_CONFIGS = [
    {
        "name": "google/gemma-2-2b",
        "seq_len": 7,
        "batch_size": 1,
    },
    {
        "name": "google/gemma-2-9b",
        "seq_len": 7,
        "batch_size": 1,
    }
]

@pytest.fixture(scope="module")
def model_modules() -> List[str]:
    """Fixture providing list of model modules to test."""
    return [
        "model.embed_tokens",
        "model.norm",
        "model.layers.0.input_layernorm",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.post_feedforward_layernorm",
        "model.layers.0.pre_feedforward_layernorm",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj"
    ]

@pytest.fixture(params=MODEL_CONFIGS)
def model_config(request) -> Dict:
    """Parametrized fixture for different model configurations."""
    return request.param

@pytest.fixture
def models(model_config: Dict) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM]:
    """Fixture providing both the converted and reference models for a given configuration."""
    # Create and save MaxText model
    max_text_model = MaxTextModel.from_preset(
        preset_handle=f"hf://{model_config['name']}",
        seq_len=model_config['seq_len'],
        per_device_batch_size=model_config['batch_size'],
        precision="float32",
        scan_layers=True,
    )
    
    checkpoint_dir = os.path.join(TMP_DIR, model_config['name'].replace('/', '_'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    max_text_model.save_in_hf_format(checkpoint_dir)
    
    # Load converted model
    converted_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, 
        torch_dtype=torch.float32
    )
    
    # Load reference model
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_config['name'], 
        torch_dtype=torch.float32
    )
    
    return converted_model, reference_model

@pytest.fixture
def test_inputs() -> dict:
    """Fixture providing test input data."""
    return {
        'input_ids': np.array([[2, 4521, 2134]]), 
        'attention_mask': np.array([[1, 1, 1]])
    }

def test_model_weights(models, model_modules, model_config):
    """Test that all model weights match between converted and reference models."""
    converted_model, reference_model = models
    
    print(f"\nTesting weights for model: {model_config['name']}")
    
    for module in model_modules:
        ref_weights = reference_model.get_submodule(module).state_dict()["weight"]
        conv_weights = converted_model.get_submodule(module).state_dict()["weight"]
        
        # Check if weights are close within tolerance
        is_close = torch.allclose(ref_weights, conv_weights, atol=0.01)
        
        if not is_close:
            # Get mismatched indices
            mismatch_mask = ~torch.isclose(ref_weights, conv_weights, atol=0.01)
            num_mismatches = mismatch_mask.sum().item()
            
            # Prepare detailed error message
            error_msg = (
                f"\nModel: {model_config['name']}"
                f"\nModule {module} weights mismatch:"
                f"\nNumber of mismatched elements: {num_mismatches}"
                f"\nReference weights (mismatched): {ref_weights[mismatch_mask]}"
                f"\nConverted weights (mismatched): {conv_weights[mismatch_mask]}"
            )
            pytest.fail(error_msg)

def test_model_logits(models, test_inputs, model_config):
    """Test that model logits match between converted and reference models."""
    converted_model, reference_model = models
    
    print(f"\nTesting logits for model: {model_config['name']}")
    
    # Get logits from both models
    converted_logits = converted_model(**test_inputs, output_hidden_states=True).logits
    reference_logits = reference_model(**test_inputs, output_hidden_states=True).logits
    
    # Check if logits are close within tolerance
    is_close = torch.allclose(converted_logits, reference_logits, atol=1.0)
    
    if not is_close:
        # Get mismatched values
        mismatch_mask = ~torch.isclose(reference_logits, converted_logits, atol=1.0)
        error_msg = (
            f"\nModel: {model_config['name']}"
            "\nLogits mismatch:"
            f"\nReference logits (mismatched): {reference_logits[mismatch_mask]}"
            f"\nConverted logits (mismatched): {converted_logits[mismatch_mask]}"
        )
        pytest.fail(error_msg)
