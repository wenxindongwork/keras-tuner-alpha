from maxtext.MaxText import pyconfig
from jax.sharding import Mesh
from maxtext.MaxText import max_utils
from maxtext.MaxText.layers.models import Transformer
from maxtext.MaxText.layers import quantizations

def get_maxtext_config(model_name="default"):
    argv = [
        "",
        "maxtext/MaxText/configs/base.yml",
        f"model_name={model_name}",
        "run_name=must_supply_but_not_needed",
    ]
    pyconfig.initialize(argv)
    config = pyconfig.config
    return config

def get_maxtext_model(maxtext_config):
    
    devices_array = max_utils.create_device_mesh(maxtext_config)
    mesh = Mesh(devices_array, maxtext_config.mesh_axes)
    quant = quantizations.configure_quantization(maxtext_config)

    # Model is not parameterized nor initialized. 
    model = Transformer(maxtext_config, mesh, quant)
    return model
