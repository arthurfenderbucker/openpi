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

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import logging
import time
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0grounded_config
from openpi.shared import array_typing as at

logger = logging.getLogger(__name__)


def _make_batch_grad_fn_from_params(guidance_params: dict):
    """Build a batched gradient function from structured guidance parameter arrays.

    Integrates x_t (SO3 velocity actions) to recover EE position trajectories,
    computes distances from those positions to the reference waypoints, then
    returns the gradient of a squared-error potential via JAX autograd.

    For each pair *i* and chunk-timestep *k*::

        pos[k]       = ee_pos_0 + cumsum(x_t[:k+1, :3])
        dist[i, k]   = distance_fn_i(pos[k], waypoints[i])   (type-dependent)
        error[i, k]  = dist[i, k] - expected[i, k]
        f(x_t)       = -0.5 * Σ_{i,k} error[i, k]²

    ``jax.grad(f)(x_t)`` propagates through the cumsum integration, yielding
    a gradient that steers each velocity step toward the reference distance
    profile at all future timesteps (causal chain effect).

    Distance type selection per pair (broadcast over T):
        - Surface (dist_type=1): min distance from pos to surface point cloud
        - Axis (has_axis=1):     dot(waypoint - pos, signed_axis)
        - Plane (has_plane=1):   ||in_plane(waypoint - pos)||
        - Centroid (default):    ||pos - waypoint||

    Axes are stored with sign encoding direction:
        - ee_is_from: +axis   →  dist = dot(waypoint - pos, axis)
        - ee_is_to:   -axis   →  dist = dot(waypoint - pos, -axis) = dot(pos - waypoint, axis)

    Args:
        guidance_params: Dict with JAX arrays (keys match ``guidance_builder.GP_KEYS``):
            - ``"__gp_ee_pos_0__"``:       (3,)      EE position at current step.
            - ``"__gp_expected_dists__"``: (N, T)    ref distances per pair per chunk step.
            - ``"__gp_dist_types__"``:     (N,)      uint8 distance type codes.
            - ``"__gp_waypoints__"``:      (N, 3)    reference centroid per pair.
            - ``"__gp_has_axis__"``:       (N,)      axis applicability flag.
            - ``"__gp_axes__"``:           (N, 3)    signed axis vectors (zeros if unused).
            - ``"__gp_has_plane__"``:      (N,)      plane applicability flag.
            - ``"__gp_plane_normals__"``:  (N, 3)    plane normals (zeros if unused).
            - ``"__gp_surface_counts__"``: (N,)      surface point counts (0 if centroid).
            - ``"__gp_surface_pts__"``:    (N, S, 3) padded surface point clouds.

    Returns:
        ``jax.vmap(jax.grad(guidance_fn))`` — same interface as the callable path.
    """
    ee_pos_0 = jnp.array(guidance_params["__gp_ee_pos_0__"])  # (3,)
    dt = jnp.array(guidance_params["__gp_dt__"])  # scalar
    expected = jnp.array(guidance_params["__gp_expected_dists__"])  # (N, T)
    dist_types = jnp.array(guidance_params["__gp_dist_types__"])  # (N,) uint8
    waypoints = jnp.array(guidance_params["__gp_waypoints__"])  # (N, 3)
    has_axis = jnp.array(guidance_params["__gp_has_axis__"])  # (N,) uint8
    axes = jnp.array(guidance_params["__gp_axes__"])  # (N, 3) signed
    has_plane = jnp.array(guidance_params["__gp_has_plane__"])  # (N,) uint8
    plane_norms = jnp.array(guidance_params["__gp_plane_normals__"])  # (N, 3)
    surf_pts = jnp.array(guidance_params["__gp_surface_pts__"])  # (N, S, 3)

    # Per-pair guidance weights. If the scalar override key is present, all
    # per-pair factors are set to that value (uniform override).
    gf_factors = jnp.array(guidance_params["__gp_guidance_factors__"])  # (N,)
    if "__gp_guidance_factor__" in guidance_params:
        gf_factors = jnp.full_like(gf_factors, guidance_params["__gp_guidance_factor__"])

    def _guidance_fn(x_t):
        # x_t: (total_horizon, action_dim) — unbatched.
        # Integrate translational velocities to get EE position at each step.
        vel = x_t[:, :3]  # (H, 3)
        positions = ee_pos_0[None, :] + jnp.cumsum(vel * dt, axis=0)  # (H, 3)

        T = expected.shape[1]
        positions = positions[:T]  # (T, 3)

        # diffs[t, i, :] = pos[t] - waypoints[i]
        diffs = positions[:, None, :] - waypoints[None, :, :]  # (T, N, 3)

        # --- Centroid distances: ||pos - waypoint|| --------------------------
        centroid_dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)  # (T, N)

        # --- Axis distances: dot(waypoint - pos, signed_axis) ----------------
        # axes is stored signed, so: dot(-diffs[t,i], axes[i]) = -dot(diffs, axes)
        axis_dists = -jnp.sum(diffs * axes[None, :, :], axis=-1)  # (T, N)

        # --- Plane distances: ||in_plane(waypoint - pos)|| -------------------
        proj = jnp.sum(diffs * plane_norms[None, :, :], axis=-1, keepdims=True)  # (T, N, 1)
        in_plane = diffs - proj * plane_norms[None, :, :]  # (T, N, 3)
        plane_dists = jnp.sqrt(jnp.sum(in_plane**2, axis=-1) + 1e-8)  # (T, N)

        # --- Surface distances: min ||pos - surf_pt|| over point cloud -------
        # positions: (T, 3), surf_pts: (N, S, 3)
        pos_exp = positions[:, None, None, :]  # (T, 1, 1, 3)
        surf_exp = surf_pts[None, :, :, :]  # (1, N, S, 3)
        surf_d2 = jnp.sum((pos_exp - surf_exp) ** 2, axis=-1)  # (T, N, S)
        surface_dists = jnp.sqrt(jnp.min(surf_d2, axis=-1) + 1e-8)  # (T, N)

        # --- Select distance per pair (broadcast flags over T) ---------------
        is_surf = (dist_types > 0)[None, :]  # (1, N)
        ha = (has_axis > 0)[None, :]  # (1, N)
        hp = (has_plane > 0)[None, :]  # (1, N)

        # dists = jnp.where(
        #     is_surf, surface_dists, jnp.where(ha, axis_dists, jnp.where(hp, plane_dists, centroid_dists))
        # )  # (T, N)

        # TEMPORALLY DISABLE SURFACE AND PLANE DISTANCES FOR TESTING:
        dists = jnp.where(
            is_surf,
            jnp.full_like(surface_dists, 0.0),  # effectively disable surface guidance
            jnp.where(
                ha,
                axis_dists,  # effectively disable surface guidance
                jnp.where(hp, jnp.full_like(plane_dists, 0.0), jnp.full_like(centroid_dists, 0.0)),
            ),
        )  # (T, N)

        errors = dists.T - expected[:, :T]  # (N, T)
        print("errors shape:", errors.shape)  # DEBUGGING

        # Weight each pair's contribution by its guidance factor.
        # gf_factors[:, None] *
        return 0.5 * jnp.sum(errors**2)

    return jax.vmap(jax.grad(_guidance_fn))


class Pi0Grounded(pi0.Pi0):
    """Pi0 model with multi-chunk action generation support."""

    def __init__(self, config: pi0grounded_config.Pi0GroundedConfig, rngs: nnx.Rngs):
        # Initialize parent Pi0 model
        super().__init__(config, rngs)
        # Store the number of chunks for inference
        self.num_action_chunks = config.num_action_chunks
        self.max_guidance_factor = 0.5

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
        if guidance_params is not None:
            num_action_chunks = guidance_params.get("__gp_num_chunks__", num_action_chunks)
            batch_grad_fn = _make_batch_grad_fn_from_params(guidance_params)
            guidance_factor = guidance_params.get("__gp_guidance_factor__", guidance_params.get("guidance_factor", 1.0))
        elif guidance_fn is not None:
            batch_grad_fn = jax.vmap(jax.grad(guidance_fn))
        else:
            batch_grad_fn = None

        debug_guidance = guidance_params.get("__gp_debug_guidance__", True)

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        # Initialize noise for all chunks
        total_horizon = num_action_chunks * self.action_horizon
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, total_horizon, self.action_dim))
        if debug_guidance:
            noise = noise * 0.0  # scale down initial noise for stability

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

            if not debug_guidance:
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
            else:
                # In debug mode, skip the forward pass and use a dummy gradient to test guidance effects in isolation.
                v_t = jnp.zeros_like(x_t)

            if batch_grad_fn is not None:
                # Classifier-style guidance: grad_{x_t} guidance_fn(x_t) steers the ODE.
                guidance_grad = batch_grad_fn(x_t)
                print("x_t shape:", x_t.shape)  # DEBUGGING
                print("guidance_grad shape:", guidance_grad.shape)  # DEBUGGING

                v_t = v_t + guidance_factor * guidance_grad

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
