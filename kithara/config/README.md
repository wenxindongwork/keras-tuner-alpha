"""Config Launcher

Supports both singlehost and multihost training.

Usage:
  Singlehost: 
    # Use default config
    python config/launcher.py

    # Use a different base config
    python config/launcher.py --config=your_base_config.yaml

    # Apply YAML overrides
    python config/launcher.py --override_config=your_override_config.yaml

    # Use command line overrides for quick experiments
    python config/launcher.py --override learning_rate=5e-5 training_steps=10000

  Multihost:  
    Wrap your singlehost command in ray/submit_job.py and pass your HuggingFace token:
    
    python ray/submit_job.py "python3.11 kithara/config/launcher.py --single_host=False --tpu_generation=v5e" --hf-token your_token

If you experience OOM error during model checkpoint loading/saving, it is because your host VM
does not have enough capacity to load/save the model. Consider mounting extra memory onto your VM,
and launch this script with:
  `HF_HOME=new_hf_cache_dir KERAS_HOME=new_keras_cache_dir python config/launcher.py`

E.g. `HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python config/launcher.py`
"""
