import ray 

def split_dataset(ds: ray.data.Dataset, num_hosts: int):
    return ds.streaming_split(num_hosts, equal=True)
