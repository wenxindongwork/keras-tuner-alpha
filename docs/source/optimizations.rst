.. _optimizations:

Performance Optimizations Supported
===================================

FlashAttention
-------------
A faster and more memory-efficient attention mechanism. Flash Attention is enabled by default for all Kithara supported models.

Remat
-----
Remat (short for Rematerialization) is a technique to reduce memory usage during training, especially beneficial for larger models. 
Rematerialization recomputes most intermediate tensors during the backward pass instead of saving them in HBM. 
This significantly reduces the HBM memory requirement. Remat is applied by default to all Kithara supported models.

Parallelism Options
------------------

FSDP
~~~~
Fully Sharded Data Parallelism (FSDP) is a technique to distribute model parameters across multiple devices. FSDP is applied to all Kithara models by default.

FSDP Explained
~~~~~~~~~~~~

1. Basic Model Structure::

       +----------------+  +----------------+  +----------------+
       |                       Layer 1                         |
       |                                                      |
       +----------------+  +----------------+  +----------------+
                                 |
                                 v
       +----------------+  +----------------+  +----------------+
       |                       Layer 2                         |
       |                                                      |
       +----------------+  +----------------+  +----------------+
                                 |
                                 v
                                ...

2. FSDP shards each layer of the model N ways onto N accelerators::

       TPU 0               TPU 1               TPU 2
       +----------------+  +----------------+  +----------------+
       |  Layer 1 (1/3) |  |  Layer 1 (2/3) |  |  Layer 1 (3/3) |
       |                |  |                |  |                |
       +----------------+  +----------------+  +----------------+
               |                  |                  |
               v                  v                  v
       +----------------+  +----------------+  +----------------+
       |  Layer 2 (1/3) |  |  Layer 2 (2/3) |  |  Layer 2 (3/3) |
       |                |  |                |  |                |
       +----------------+  +----------------+  +----------------+

3. However, we need the complete layer to carry out the computation,
   meaning we need to gather the weights before running the input through a layer::

       TPU 0               TPU 1               TPU 2
       +----------------+  +----------------+  +----------------+
       |                       Input Batch                      |
       |            Replicated on all TPU devices               |
       +----------------+  +----------------+  +----------------+
               |                  |                  |
               v                  v                  v
       +----------------+  +----------------+  +----------------+
       |  Layer 1 (1/3) |  |  Layer 1 (2/3) |  |  Layer 1 (3/3) |
       |                |  |                |  |                |
       +----------------+  +----------------+  +----------------+
               |                  |                  |
               v                  v                  v
       +----------------+  +----------------+  +----------------+
       | All-to-All     <==================================>    |
       | Communication                                          |
       +----------------+  +----------------+  +----------------+
               |                  |                  |
               v                  v                  v
       +----------------+  +----------------+  +----------------+
       |                       Layer 1                          |
       |            Replicated on all TPU devices               |
       +----------------+  +----------------+  +----------------+
                               |
                               v
       +----------------+  +----------------+  +----------------+
       |                       Result                           |
       |               Replicated on all TPU devices            |
       +----------------+  +----------------+  +----------------+

4. Right now, each TPU is doing the exact same work! It is processing
   the same batch of data, and getting the exact same result.
   Seems like there is room for optimization here!
   Why not let each TPU process different inputs? That way,
   we can process a larger batch all together::

       +----------------+  +----------------+  +----------------+
       | Input Batch  1 |  | Input Batch 2  |  | Input Batch 3  |
       |                |  |                |  |                |
       +----------------+  +----------------+  +----------------+
               |                  |                  |
               v                  v                  v
       +----------------+  +----------------+  +----------------+
       |  Layer 1 (1/3) |  |  Layer 1 (2/3) |  |  Layer 1 (3/3) |
       |                |  |                |  |                |
       +----------------+  +----------------+  +----------------+
               |                  |                  |
               v                  v                  v
       +----------------+  +----------------+  +----------------+
       | All-to-All     <==================================>    |
       | Communication                                          |
       +----------------+  +----------------+  +----------------+
               |                  |                  |
               v                  v                  v
       +----------------+  +----------------+  +----------------+
       |                       Layer 1                          |
       |              Replicated on all TPU devices             |
       +----------------+  +----------------+  +----------------+
                               |
                               v
       +----------------+  +----------------+  +----------------+
       |  Result 1      |  |  Result 2      |  |  Result 3      |
       |                |  |                |  |                |
       +----------------+  +----------------+  +----------------+

And that is FSDP!