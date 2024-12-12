from keras_nlp.src.utils.transformers.safetensor_utils import SafetensorLoader
from keras_nlp.src.utils.preset_utils import load_json
import numpy as np

def print_maxtext_model_variables(model):
    for variable in model.weights:
        print(variable.path, variable.shape)


def print_hg_safetensor_weight_keys(preset_handle):
    with SafetensorLoader(preset_handle) as loader:
        weight_map_keys = loader.safetensor_config["weight_map"].keys()
        print(weight_map_keys)


def get_maxtext_model_type_from_hf_handle(preset_handle):
    config = load_json(preset_handle)
    model_type = config["model_type"]
    if model_type == "gemma2":
        n_layers = config["num_hidden_layers"]
        if n_layers == 26:
            return "gemma2-2b"
        elif n_layers == 42:
            return "gemma2-9b"
        elif n_layers == 46:
            return "gemma2-27b"
    print(f"model type {model_type} is currently unsupported.")
    return None


def match_tensor_shape(A, target_shape):
    """
    Reshape and permute tensor A to match the target shape if possible.

    Parameters:
    A (np.ndarray): Input tensor to be reshaped/permuted
    target_shape (tuple): Target shape to match

    Returns:
    np.ndarray: Reshaped and permuted version of A that matches target shape

    Raises:
    ValueError: If shapes cannot be matched through reshaping and permutation
    """
    # Get shapes and total elements
    shape_A = A.shape
    total_elements_A = np.prod(shape_A)
    total_elements_target = np.prod(target_shape)

    # Check if total elements match
    if total_elements_A != total_elements_target:
        raise ValueError(
            f"Cannot reshape: total elements don't match. Input has {total_elements_A}, target has {total_elements_target}")

    # If shapes are identical, try direct permutation
    if sorted(shape_A) == sorted(target_shape):
        permutation = []
        for dim_target in target_shape:
            for i, dim_A in enumerate(shape_A):
                if dim_A == dim_target and i not in permutation:
                    permutation.append(i)
                    break
        return np.transpose(A, permutation)

    # If shapes differ but elements match, try smart reshaping
    else:
        # First try to identify common dimensions between shapes
        common_dims = []
        remaining_target_dims = list(target_shape)
        remaining_A_dims = list(shape_A)

        # Find matching dimensions
        for dim_A in shape_A:
            if dim_A in remaining_target_dims:
                common_dims.append(dim_A)
                remaining_target_dims.remove(dim_A)
                remaining_A_dims.remove(dim_A)

        # For example: (2048, 3584) -> (3584, 8, 256)
        # We know 3584 is common, so we need to reshape 2048 into (8, 256)

        # If we found common dimensions, use them as anchor points
        if common_dims:
            # Create intermediate shape that preserves common dimensions
            intermediate_shape = []
            remaining_elements = total_elements_A

            # Build up the target shape using common dimensions as anchors
            for dim in target_shape:
                if dim in common_dims:
                    intermediate_shape.append(dim)
                    remaining_elements //= dim

            # Add remaining elements as a single dimension
            if remaining_elements > 1:
                intermediate_shape.append(remaining_elements)

            # Reshape and then transpose to final shape
            try:
                # First reshape to intermediate shape
                reshaped = A.reshape(intermediate_shape)

                # If the shapes still don't match, do final reshape
                if reshaped.shape != target_shape:
                    reshaped = reshaped.reshape(target_shape)

                return reshaped

            except ValueError as e:
                raise ValueError(
                    f"Cannot reshape tensor from {shape_A} to {target_shape}: {str(e)}")

        # If no common dimensions, try direct reshape
        try:
            return A.reshape(target_shape)
        except ValueError as e:
            raise ValueError(
                f"Cannot reshape tensor from {shape_A} to {target_shape}: {str(e)}")
