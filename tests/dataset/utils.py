"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import ray 
from datasets import load_dataset

def create_dict_ray_dataset(n=1000):    
    dataset_items = [
        {"text": f"{i} What is your name? My name is Kithara."} for i in range(n)
    ]
    return ray.data.from_items(dataset_items)

def create_hf_streaming_ray_dataset():    
    hf_val_dataset = load_dataset(
        "allenai/c4", "en", split="validation", streaming=True
    )
    return ray.data.from_huggingface(hf_val_dataset)

def create_json_ray_dataset():    
    ds = ray.data.read_json("s3:/import ray/anonymous@ray-example-data/log.json")
    return ds

def create_csv_ray_dataset():
    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
    return ds