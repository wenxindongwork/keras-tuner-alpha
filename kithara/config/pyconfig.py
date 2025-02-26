# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from typing import Any
import yaml
import argparse
import pprint

DEFAULT_CONFIG = "kithara/config/default.yaml"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Config loader with overrides")
    
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to base config YAML file")
    
    parser.add_argument("--override_config", type=str, default=None,
                        help="Optional path to YAML file with config overrides")
    
    parser.add_argument("--override", nargs="*", default=[],
                        help="Command line overrides in key=value format")
    
    args = parser.parse_args()
    
    # Process command line overrides
    override_dict = {}
    for arg in args.override:
        if "=" in arg:
            key, value = arg.split("=", 1)
            override_dict[key] = value
            print("key", key, type(value))
    
    args.override_dict = override_dict
    
    return args


def load_config() -> dict[str, Any]:
    """Loads the YAML config from a file with a given name."""
  
    # Parse command line arguments
    args = parse_args()

    # Load base config
    config_file = args.config
    with open(config_file, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)
        

  # Apply YAML file overrides
    if args.override_config is not None:
        with open(args.override_config, "r", encoding="utf-8") as override_file:
            override_config = yaml.safe_load(override_file)
            for key, value in override_config.items():
                config[key] = value

    # Apply command line overrides
    if args.override_dict is not None:
        for key, value in args.override_dict.items():
            config[key] = value
    
    cast_numerical_type(config)
    pprint.pprint(config)
    return config

def cast_numerical_type(dictionary):
    for key, value in dictionary.items():
        try:
            # First try as int
            value = int(value)
        except Exception :
            try:
                # Then as float
                value = float(value)
            except Exception:
                try: 
                    # Check for boolean values
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                except Exception:
                    # Otherwise, keep as string
                    continue   
        dictionary[key] = value
    return dictionary