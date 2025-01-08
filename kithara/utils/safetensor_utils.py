from typing import Optional, Union, List, Dict, Tuple, Optional, Any
from jaxtyping import Array

SAFE_TENSORS_MODEL = "model.safetensors"
SAFE_TENSORS_INDEX_NAME = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE = 1024 * 1024 * 1024 * 3  # 3GB default


def shard_checkpoint(
    weights_dict: Dict[str, Array],
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    weights_name: str = "model.safetensors",
) -> Tuple[Dict[str, Dict[str, Array]], Optional[Dict]]:
    """Shards a model checkpoint into smaller pieces based on size constraints.

    Args:
        weights_dict: Model weights dictionary to shard
        max_shard_size: Maximum size in bytes for each shard
        weights_name: Base filename for the shards

    Returns:
        Tuple of (sharded weights dict, optional index dict)
        Index contains metadata and weight mapping information
    """
    # Track current shard and accumulated sizes
    current_shard: Dict[str, Array] = {}
    shards: List[Dict[str, Array]] = [current_shard]
    current_size = 0
    total_size = 0

    # Iterate through weights in sorted order for deterministic sharding
    for key, tensor in sorted(weights_dict.items()):
        weight_size = tensor.numel() * tensor.itemsize
        # Start new shard if current one would exceed size limit
        if (current_size + weight_size > max_shard_size) and len(current_shard.items()):
            current_shard = {}
            shards.append(current_shard)
            current_size = 0

        # Add weight to current shard and update sizes
        current_shard[key] = tensor
        current_size += weight_size
        total_size += weight_size

    # Return single shard without index if no sharding needed
    if len(shards) == 1:
        return {weights_name: shards[0]}, None

    # Generate shard filenames and build index
    shard_dict = {}
    weight_map = {}

    for idx, shard in enumerate(shards, 1):
        # Create numbered shard filename
        shard_name = weights_name.replace(
            ".safetensors", f"-{idx:05d}-of-{len(shards):05d}.safetensors"
        )
        shard_dict[shard_name] = shard

        # Map each weight to its shard file
        for key in shard:
            weight_map[key] = shard_name

    return shard_dict, {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
