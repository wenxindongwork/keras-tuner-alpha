import jax
import platform


def get_device_stats():
    device_count = jax.device_count()
    device_kind = jax.devices()[0].device_kind
    memory_info = jax.devices()[0].memory_stats()
    memory_per_device = memory_info["bytes_limit"] / (1024**3)  # Convert to GB
    total_memory = memory_per_device * device_count
    return {
        "device_count": device_count,
        "device_kind": device_kind,
        "memory_per_device_gb": round(memory_per_device, 2),
        "total_memory_gb": round(total_memory, 2),
    }


def print_kithara_logo_and_platform_info():
    platform_system = platform.system()
    tpu_stats = get_device_stats()
    statistics = (
        f"       '==='\n"
        f"        |||\n"
        f"     '- ||| -'\n"
        f"    /  |||||  \\   Kithara. Platform: {platform_system}. JAX: {jax.__version__}\n"
        f"   |   (|||)   |  Hardware: {tpu_stats['device_kind']}. Device count: {tpu_stats['device_count']}.\n"
        f"   |   |◕‿◕|   |  HBM Per Device: {tpu_stats['memory_per_device_gb']} GB. Total HBM Memory: {tpu_stats['total_memory_gb']} GB\n"
        f"    \\  |||||  /   Free Apache license: http://github.com/ai-hypercomputer/kithara\n"
        f"     --|===|--"
    )
    print(statistics)
