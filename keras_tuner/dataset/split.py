import ray 

def split_ray_dataset(ds: ray.data.Dataset, num_hosts: int):
    return ds.streaming_split(num_hosts, equal=True)


# def split_files(files: List[str], num_hosts:int):
#     if len(files)>