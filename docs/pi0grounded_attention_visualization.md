# Pi0Grounded Attention Mask Visualization

## Independent Chunks with Block-Diagonal Attention

This document visualizes how the attention masks ensure that each chunk is truly independent.

### Example: 3 Chunks, 4 Steps Each

```
Prefix tokens: P1, P2, P3
Chunk 1 tokens: C1_0, C1_1, C1_2, C1_3
Chunk 2 tokens: C2_0, C2_1, C2_2, C2_3
Chunk 3 tokens: C3_0, C3_1, C3_2, C3_3
```

### Attention Pattern

Each token can attend to:

**Chunk 1:**
- C1_0 can attend to: [P1, P2, P3, C1_0]
- C1_1 can attend to: [P1, P2, P3, C1_0, C1_1]
- C1_2 can attend to: [P1, P2, P3, C1_0, C1_1, C1_2]
- C1_3 can attend to: [P1, P2, P3, C1_0, C1_1, C1_2, C1_3]

**Chunk 2:**
- C2_0 can attend to: [P1, P2, P3, C2_0] ← Only prefix + itself!
- C2_1 can attend to: [P1, P2, P3, C2_0, C2_1]
- C2_2 can attend to: [P1, P2, P3, C2_0, C2_1, C2_2]
- C2_3 can attend to: [P1, P2, P3, C2_0, C2_1, C2_2, C2_3]

**Chunk 3:**
- C3_0 can attend to: [P1, P2, P3, C3_0] ← Only prefix + itself!
- C3_1 can attend to: [P1, P2, P3, C3_0, C3_1]
- C3_2 can attend to: [P1, P2, P3, C3_0, C3_1, C3_2]
- C3_3 can attend to: [P1, P2, P3, C3_0, C3_1, C3_2, C3_3]

### Key Properties

1. **No cross-chunk attention**: C2_0 cannot see C1_0, C1_1, C1_2, C1_3
2. **Shared prefix**: All chunks see the same observation (prefix tokens)
3. **Causal within chunk**: Later tokens in a chunk see earlier tokens in the same chunk
4. **Different noise**: Each chunk starts with different random noise at t=1

### Attention Mask Matrix

```
           P1 P2 P3 | C1_0 C1_1 C1_2 C1_3 | C2_0 C2_1 C2_2 C2_3 | C3_0 C3_1 C3_2 C3_3
          -----------------------------------------------------------------------
C1_0    |  1  1  1  |  1    0    0    0    |  0    0    0    0    |  0    0    0    0
C1_1    |  1  1  1  |  1    1    0    0    |  0    0    0    0    |  0    0    0    0
C1_2    |  1  1  1  |  1    1    1    0    |  0    0    0    0    |  0    0    0    0
C1_3    |  1  1  1  |  1    1    1    1    |  0    0    0    0    |  0    0    0    0
          -----------------------------------------------------------------------
C2_0    |  1  1  1  |  0    0    0    0    |  1    0    0    0    |  0    0    0    0
C2_1    |  1  1  1  |  0    0    0    0    |  1    1    0    0    |  0    0    0    0
C2_2    |  1  1  1  |  0    0    0    0    |  1    1    1    0    |  0    0    0    0
C2_3    |  1  1  1  |  0    0    0    0    |  1    1    1    1    |  0    0    0    0
          -----------------------------------------------------------------------
C3_0    |  1  1  1  |  0    0    0    0    |  0    0    0    0    |  1    0    0    0
C3_1    |  1  1  1  |  0    0    0    0    |  0    0    0    0    |  1    1    0    0
C3_2    |  1  1  1  |  0    0    0    0    |  0    0    0    0    |  1    1    1    0
C3_3    |  1  1  1  |  0    0    0    0    |  0    0    0    0    |  1    1    1    1
```

This is a **block-diagonal causal mask** where:
- All chunks can attend to the prefix (leftmost 3 columns all 1s)
- Each chunk forms a lower-triangular block on the diagonal
- Off-diagonal blocks are all 0s (no cross-chunk attention)

### Equivalence to Parallel Queries

This single forward pass with block-diagonal attention is equivalent to:

```python
# Conceptually (inefficient):
chunks = []
for i in range(num_chunks):
    noise_i = random_noise[i]  # Different noise for each chunk
    chunk_i = action_expert(observation, noise_i, timestep)
    chunks.append(chunk_i)
result = stack(chunks)  # Shape: [num_chunks, action_horizon, action_dim]

# Actually (efficient):
all_noise = concatenate([random_noise[0], random_noise[1], random_noise[2]])
result = action_expert_with_block_diagonal_mask(observation, all_noise, timestep)
result = reshape(result, [num_chunks, action_horizon, action_dim])
```

### Denoising Process

At each denoising timestep t:
1. All chunks have the same timestep value but different noisy states x_t
2. Each chunk independently predicts its velocity v_t based on:
   - The shared observation (prefix)
   - Its own current noisy state
   - Its own causal history
3. Each chunk updates independently: x_{t+dt} = x_t + dt * v_t
4. This repeats until t=0, producing num_chunks different action sequences

This ensures that the multiple chunks represent **diverse possible action sequences** conditioned on the same observation, not a single long concatenated sequence.
