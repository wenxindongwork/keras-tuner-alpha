from jax.sharding import NamedSharding
from jax.tree_util import tree_reduce
from jax.tree_util import tree_map

def is_not_sharded(jax_array):
    if isinstance(jax_array.sharding, NamedSharding):
        return all(x is None for x in jax_array.sharding.spec)
    return False

def is_not_sharded_and_is_large(jax_array, threshold= 1000*20): #20 MB
    if isinstance(jax_array.sharding, NamedSharding):
        return all(x is None for x in jax_array.sharding.spec) and jax_array.size >= threshold
    return False

def any_not_sharded_pytree(pytree):
    return tree_reduce(lambda x, y: x or y, tree_map(is_not_sharded, pytree))

def get_size_mb(jax_array):
    bytes_size = jax_array.nbytes   
    return int(bytes_size / (1024 * 1024))
