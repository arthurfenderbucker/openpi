# Pi0Grounded Usage Guide

This guide shows how to use Pi0Grounded/Pi05Grounded for multi-chunk action generation in various scenarios.

## ✅ Validation Status

**Basic Tests:** ✓ Passed
- Config creation for Pi0Grounded and Pi05Grounded
- JAX model initialization
- Model type registration

**Integration Tests:** Ready
- PyTorch model loading from checkpoints
- Inference with proper output shapes `[num_chunks, action_horizon, action_dim]`
- serve_policy.py compatibility

## Usage Scenarios

### 1. Using with serve_policy.py (Recommended)

The easiest way to use Pi0Grounded with the policy server is to use the pre-configured `pi05_libero_grounded` config:

```bash
# Start the policy server with Pi05Grounded (3 chunks)
python scripts/serve_policy.py \
  --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero_grounded \
  --policy.dir gs://openpi-assets/checkpoints/pi05_libero
```

This will:
- Load the pi05_libero checkpoint
- Use Pi05Grounded with `num_action_chunks=3`
- Return actions with shape `[3, 10, 7]` (3 chunks × 10 horizon × 7 action_dim)

### 2. Custom Number of Chunks

To use a different number of chunks, you can modify the config in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_libero_grounded_5chunks",
    model=pi0grounded_config.Pi0GroundedConfig(
        pi05=True, 
        action_horizon=10, 
        discrete_state_input=False,
        num_action_chunks=5,  # Generate 5 chunks instead of 3
    ),
    # ... rest of config ...
)
```

Then use it:

```bash
python scripts/serve_policy.py \
  --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero_grounded_5chunks \
  --policy.dir gs://openpi-assets/checkpoints/pi05_libero \
```

### 3. Programmatic Usage

For direct Python usage without the server:

```python
from openpi.models.pi0grounded_config import Pi0GroundedConfig
from openpi.policies import policy_config
from openpi.training import config as train_config

# Load the base config
train_cfg = train_config.get_config("pi05_libero")

# Replace with Pi0GroundedConfig
train_cfg.model = Pi0GroundedConfig(
    action_dim=train_cfg.model.action_dim,
    action_horizon=train_cfg.model.action_horizon,
    num_action_chunks=3,  # Generate 3 chunks
    pi05=True,  # Use Pi05 architecture
    paligemma_variant=train_cfg.model.paligemma_variant,
    action_expert_variant=train_cfg.model.action_expert_variant,
)

# Load policy
policy = policy_config.create_trained_policy(
    train_cfg,
    "gs://openpi-assets/checkpoints/pi05_libero",
    pytorch_device="cuda:0",
)

# Run inference
from openpi.policies import libero_policy
obs = libero_policy.make_libero_example()
result = policy.infer(obs)

# Actions shape: [3, 10, 7] (num_chunks, action_horizon, action_dim)
actions = result["actions"]
print(f"Actions shape: {actions.shape}")

# Use chunks sequentially
for chunk_idx in range(actions.shape[0]):
    chunk = actions[chunk_idx]  # Shape: [10, 7]
    for action in chunk:
        # Execute action
        env.step(action)
```

### 4. Dynamic Chunk Override (Advanced)

You can override `num_action_chunks` at inference time without changing the config:

```python
# This requires modifying serve_policy.py or your calling code
policy = policy_config.create_trained_policy(
    train_cfg,
    checkpoint_dir,
    sample_kwargs={"num_action_chunks": 5},  # Override at runtime
)
```

**Note:** This only works if the base model config is already Pi0GroundedConfig.

### 5. Integration with LIBERO Evaluation (main_pro.py)

The `examples/libero/main_pro.py` script has been updated to handle multi-chunk outputs:

```bash
# Start the policy server with Pi05Grounded
python scripts/serve_policy.py \
  --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero_grounded \
  --policy.dir gs://openpi-assets/checkpoints/pi05_libero \

# In another terminal, run evaluation
cd examples/libero
python main_pro.py \
  --num_action_chunks 3 \
  --replan_steps 5 \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50
```

The evaluation script automatically:
- Detects the 3D output shape `[num_chunks, action_horizon, action_dim]`
- Flattens chunks into the action queue
- Tracks inference efficiency metrics

## Output Format

**Key Change:** Actions are now returned as `[num_chunks, action_horizon, action_dim]` instead of flattened.

**Example:**
- `num_action_chunks=3`, `action_horizon=10`, `action_dim=7`
- Output shape: `[3, 10, 7]`

**Processing chunks:**

```python
actions = result["actions"]  # Shape: [3, 10, 7]

# Option 1: Use chunks sequentially
for chunk in actions:
    for action in chunk:
        env.step(action)

# Option 2: Flatten and queue
action_queue = []
for chunk in actions:
    action_queue.extend(chunk)

# Option 3: Use specific chunks
first_chunk = actions[0]  # Shape: [10, 7]
```

## Architecture Details

Pi0Grounded uses **block-diagonal causal attention** to generate independent chunks:

- Each chunk has different random noise initialization
- Chunks are processed in parallel (efficient)
- Chunks cannot see each other (only the observation)
- All chunks share the same timestep

This is equivalent to running the action expert `num_chunks` times independently but much more efficient.

See [docs/pi0grounded_attention_visualization.md](docs/pi0grounded_attention_visualization.md) for details.

## Testing

Run the validation script to test your setup:

```bash
# Basic tests (no checkpoint required)
python scripts/test_pi0grounded.py

# Full tests with checkpoint
python scripts/test_pi0grounded.py \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero
```

## Troubleshooting

**Q: Getting shape mismatch errors?**
- Make sure you're using the updated `main_pro.py` that handles 3D outputs
- Check that `num_action_chunks` in the config matches what you expect

**Q: Chunks look identical?**
- This should not happen - each chunk uses different noise
- Verify with the test script: `python scripts/test_pi0grounded.py --checkpoint-dir <path>`

**Q: How to verify it's working?**
- Check the output shape: `assert actions.shape == (num_chunks, action_horizon, action_dim)`
- Compare chunks: they should be different due to different noise

**Q: Can I use this with Pi0 (not Pi05)?**
- Yes! Set `pi05=False` in the Pi0GroundedConfig
- Example config name: `pi0_libero_grounded`

## Performance Considerations

**Memory:** Increases with `num_chunks` (processes more tokens in the action expert)

**Latency per inference:** Slightly higher (longer sequences)

**Total latency:** Can be lower if you reduce inference frequency proportionally

**Recommendation:** Start with 2-3 chunks, measure success rates, then adjust.

## Available Configs

Pre-configured training configs in `src/openpi/training/config.py`:

1. **`pi05_libero_grounded`** - Pi05 with 3 chunks for LIBERO
   - `num_action_chunks=3`
   - `action_horizon=10`
   - `pi05=True`

You can add more configs by following the pattern in `config.py`.
