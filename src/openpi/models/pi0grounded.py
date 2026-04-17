"""Pi0Grounded: Extended Pi0 model with multi-chunk action generation.

This model extends Pi0 to support generating multiple independent action chunks
in a single inference call, reducing the frequency of expensive model queries during
deployment.

Key features:
- Generates num_action_chunks independent action sequences per inference
- Each chunk uses different random noise initialization
- Chunks are processed in parallel using block-diagonal causal attention
- Each chunk can only attend to the prefix (observation) and itself, not other chunks
- This is equivalent to running the action expert num_chunks times independently,
  but more computationally efficient
- Backward compatible: num_action_chunks=1 matches original Pi0 exactly
"""

import logging
import time

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from policy_steering import build_guidance_fns
from policy_steering import compute_guidance_velocity
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0grounded_config
from openpi.shared import array_typing as at

logger = logging.getLogger(__name__)


class Pi0Grounded(pi0.Pi0):
    """Pi0 model with multi-chunk action generation support."""

    def __init__(self, config: pi0grounded_config.Pi0GroundedConfig, rngs: nnx.Rngs):
        # Initialize parent Pi0 model
        super().__init__(config, rngs)
        # Store the number of chunks for inference
        self.num_action_chunks = config.num_action_chunks
        # Optional external JAX dynamics function for guidance
        self._dynamics_fn = None

    def set_dynamics_fn(self, fn):
        """Set a JAX-differentiable dynamics function for guidance.

        The function signature must be::

            integrate_actions(x_t, ee_pos_0, ee_ori_0, dt)
                -> (positions, orientations, gripper)

        where all inputs/outputs are JAX arrays.
        """
        self._dynamics_fn = fn

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        num_chunks: int | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed suffix with support for multiple action chunks.

        Args:
            obs: Observation data
            noisy_actions: Noisy action sequence, shape [b, num_chunks * action_horizon, action_dim]
            timestep: Flow matching timestep
            num_chunks: Number of action chunks (if None, uses self.num_action_chunks)

        Returns:
            Embedded tokens, masks, AR mask, and optional adaRMS conditioning
        """
        if num_chunks is None:
            num_chunks = self.num_action_chunks

        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # add a single state token (shared across all chunks)
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding
        time_emb = pi0.posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # Expand time embedding for all action steps across all chunks
            total_steps = num_chunks * self.action_horizon
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=total_steps)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        # Create AR mask for multiple chunks
        # Each chunk: first token breaks causality (True), rest are causal (False)
        for _ in range(num_chunks):
            ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        num_action_chunks: int | None = None,
        guidance_fn=None,
        guidance_params: dict | None = None,
        guidance_factor: float = 0.0,
    ) -> _model.Actions:
        """Sample actions with support for multiple chunks.

        Args:
            rng: Random key for sampling
            observation: Input observation
            num_steps: Number of denoising steps
            noise: Optional initial noise
            num_action_chunks: Number of chunks to generate (overrides config if provided)

        Returns:
            Actions with shape [batch_size, num_chunks, action_horizon, action_dim]
        """
        if num_action_chunks is None:
            num_action_chunks = self.num_action_chunks

        observation = _model.preprocess_observation(None, observation, train=False)

        # Hoist outside the loop to avoid retracing jax.grad on every denoising step.
        # guidance_params (plain arrays) takes priority and keeps JIT active;
        # guidance_fn (callable) is the legacy path that bypasses module_jit.
        batch_loss_fn = None
        batch_grad_fn = None
        if guidance_params is not None:
            num_action_chunks = guidance_params.get("__gp_num_chunks__", num_action_chunks)
            # Truncate expected distances to first chunk only — guidance is
            # computed exclusively on the first action chunk.
            if "__gp_expected_dists__" in guidance_params:
                guidance_params = dict(guidance_params)
                guidance_params["__gp_expected_dists__"] = guidance_params["__gp_expected_dists__"][
                    :, : self.action_horizon
                ]
                batch_loss_fn, batch_grad_fn = build_guidance_fns(guidance_params, dynamics_fn=self._dynamics_fn)
            guidance_factor = guidance_params.get("__gp_guidance_factor__", guidance_params.get("guidance_factor", 1.0))
        elif guidance_fn is not None:
            batch_grad_fn = jax.vmap(jax.grad(guidance_fn))
            batch_loss_fn = jax.vmap(guidance_fn)

        use_line_search = guidance_params.get("__gp_line_search__", True) if guidance_params is not None else False
        debug_guidance = guidance_params.get("__gp_debug_guidance__", False) if guidance_params is not None else False

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        # Initialize noise for all chunks
        total_horizon = num_action_chunks * self.action_horizon
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, total_horizon, self.action_dim))
        noise = jnp.where(debug_guidance, noise * 0.0, noise)  # scale down initial noise for stability in debug mode

        # first fill KV cache with a forward pass of the prefix (unchanged)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # Embed suffix with multiple chunks
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), num_chunks=num_action_chunks
            )

            # Create block-diagonal causal attention mask for independent chunks
            # Each chunk can only attend to itself (and prefix), not to other chunks
            suffix_len = suffix_tokens.shape[1]
            suffix_attn_mask = jnp.zeros((batch_size, suffix_len, suffix_len), dtype=jnp.bool_)

            for chunk_idx in range(num_action_chunks):
                start_idx = chunk_idx * self.action_horizon
                end_idx = start_idx + self.action_horizon

                # Create causal mask for this chunk
                chunk_mask = jnp.tril(jnp.ones((self.action_horizon, self.action_horizon), dtype=jnp.bool_))

                # Place it in the block-diagonal position
                suffix_attn_mask = suffix_attn_mask.at[:, start_idx:end_idx, start_idx:end_idx].set(
                    jnp.broadcast_to(chunk_mask[None, :, :], (batch_size, self.action_horizon, self.action_horizon))
                )

            # Apply suffix_mask to handle padding
            valid_mask = suffix_mask[:, None, :] * suffix_mask[:, :, None]
            suffix_attn_mask = jnp.logical_and(suffix_attn_mask, valid_mask)

            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens (all suffix tokens can attend to all prefix tokens)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            # Extract all action predictions (for all chunks)
            v_t = self.action_out_proj(suffix_out[:, -total_horizon:])
            # In debug mode, zero out the forward pass result to test guidance effects in isolation.
            v_t = jnp.where(debug_guidance, jnp.zeros_like(v_t), v_t)

            if batch_grad_fn is not None:
                v_t = compute_guidance_velocity(
                    batch_loss_fn,
                    batch_grad_fn,
                    x_t,
                    v_t,
                    dt,
                    self.action_horizon,
                    guidance_factor,
                    use_line_search,
                )

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        # Start timing
        start_time = time.time()

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        # Block until computation is complete and measure time
        # x_0 = jax.block_until_ready(x_0)
        inference_time = time.time() - start_time
        logger.info(
            f"Action chunk generation took {inference_time:.4f} seconds for {num_action_chunks} chunks with {num_steps} denoising steps"
        )

        # Reshape to [batch, num_chunks, action_horizon, action_dim]
        x_0 = x_0.reshape(batch_size, self.num_action_chunks, self.action_horizon, self.action_dim)
        return x_0
