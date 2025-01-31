import ray 
from datasets import load_dataset

def create_dict_ray_dataset():    
    dataset_items = [
        {"text": f"{i} What is your name? My name is Kithara."} for i in range(1000)
    ]
    return ray.data.from_items(dataset_items)

def create_hf_streaming_ray_dataset():    
    hf_val_dataset = load_dataset(
        "allenai/c4", "en", split="validation", streaming=True
    )
    return ray.data.from_huggingface(hf_val_dataset)

def create_json_ray_dataset():    
    ds = ray.data.read_json("s3://anonymous@ray-example-data/log.json")
    return ds

def create_csv_ray_dataset():
    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
    return ds