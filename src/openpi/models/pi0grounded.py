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


# --------------------------------------------------------------------------- #
# Panda gripper geometry constants (metres) — must match utils/gripper.py     #
# --------------------------------------------------------------------------- #
_FINGER_X_REST = 0.0085
_FINGER_Z_FROM_GRIP = 0.0114


def _jax_axis_angle_to_matrix(rotvec):
    """Convert axis-angle vector (3,) to rotation matrix (3, 3) via Rodrigues.

    Handles the near-zero-rotation case by adding a small epsilon to the angle
    norm so that the division is safe and gradients remain finite.
    """
    angle = jnp.linalg.norm(rotvec)
    axis = rotvec / (angle + 1e-8)
    K = jnp.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return jnp.eye(3) + jnp.sin(angle) * K + (1.0 - jnp.cos(angle)) * (K @ K)


def _jax_finger_positions(ee_pos, ee_ori, gripper_val):
    """Compute both finger-tip positions in world frame using JAX ops.

    Uses the symmetric approximation ``q1 = gripper_val / 2``,
    ``q2 = -gripper_val / 2`` (valid for the Panda gripper where
    ``gripper_val = q1 - q2`` and the fingers are symmetric).

    Args:
        ee_pos: (3,) EE grip-site position in world frame.
        ee_ori: (3,) EE orientation as axis-angle vector.
        gripper_val: scalar gripper opening (q1 - q2).

    Returns:
        (left_tip, right_tip) — each (3,) in world frame.
    """
    q1 = gripper_val / 2.0
    q2 = -gripper_val / 2.0
    left_local = jnp.array([0.0, -(q1 + _FINGER_X_REST), _FINGER_Z_FROM_GRIP])
    right_local = jnp.array([0.0, -q2 + _FINGER_X_REST, _FINGER_Z_FROM_GRIP])
    R = _jax_axis_angle_to_matrix(ee_ori)
    return ee_pos + R @ left_local, ee_pos + R @ right_local


# Vectorized version over time dimension
_jax_finger_positions_vmap = jax.vmap(_jax_finger_positions, in_axes=(0, 0, 0), out_axes=(0, 0))


def _make_batch_guidance_fns_from_params(guidance_params: dict, dynamics_fn=None):
    """Build batched loss AND gradient functions from structured guidance parameter arrays.

    Integrates x_t (SO3 velocity actions) to recover EE state trajectories,
    computes distances from those states (or derived quantities like finger
    positions) to the reference waypoints, then returns both a loss and a
    gradient of a squared-error potential via JAX autograd.

    Source entity types (per pair, selected by ``__gp_source_type__``):
        - 0 (EE):        distance from EE position
        - 1 (finger_l):  distance from left finger tip (via FK)
        - 2 (finger_r):  distance from right finger tip (via FK)
        - 3 (orientation): orientation projection ``dot(ori, axis)``
        - 4 (gripper):    gripper aperture scalar

    For source types 0–2, the distance computation is selected by
    ``dist_type`` (centroid/axis/plane/surface).  For types 3–4, dedicated
    distance semantics apply.

    Args:
        guidance_params: Dict with JAX arrays (keys match ``guidance_builder.GP_KEYS``).
        dynamics_fn: Optional JAX-differentiable dynamics function with signature
            ``integrate_actions(x_t, ee_pos_0, ee_ori_0, dt) -> (positions, orientations, gripper)``.
            When ``None``, the default Euler integration (cumsum) is used.

    Returns:
        (batch_loss_fn, batch_grad_fn) where:
        - batch_loss_fn(x_t) → (B,) per-sample losses
        - batch_grad_fn(x_t) → (B, T, D) per-sample gradients
    """
    ee_pos_0 = jnp.array(guidance_params["__gp_ee_pos_0__"])  # (3,)
    dt = jnp.array(guidance_params["__gp_dt__"])  # scalar
    expected = jnp.array(guidance_params["__gp_expected_dists__"])  # (N, T)
    dist_types = jnp.array(guidance_params["__gp_dist_types__"])  # (N,) uint8
    source_types = jnp.array(
        guidance_params.get(
            "__gp_source_type__",
            jnp.zeros(expected.shape[0], dtype=jnp.uint8),
        )
    )  # (N,) uint8
    waypoints = jnp.array(guidance_params["__gp_waypoints__"])  # (N, 3)
    has_axis = jnp.array(guidance_params["__gp_has_axis__"])  # (N,) uint8
    axes = jnp.array(guidance_params["__gp_axes__"])  # (N, 3) signed
    has_plane = jnp.array(guidance_params["__gp_has_plane__"])  # (N,) uint8
    plane_norms = jnp.array(guidance_params["__gp_plane_normals__"])  # (N, 3)
    surf_pts = jnp.array(guidance_params["__gp_surface_pts__"])  # (N, S, 3)
    surf_counts = jnp.array(guidance_params["__gp_surface_counts__"])  # (N,) int32

    gf_weights = jnp.array(guidance_params["__gp_guidance_weights__"])  # (N,)
    if "__gp_guidance_factor__" in guidance_params:
        gf_weights = jnp.full_like(gf_weights, guidance_params["__gp_guidance_factor__"])

    # EE orientation and gripper state for FK-based guidance
    ee_ori_0 = jnp.array(guidance_params.get("__gp_ee_ori_0__", jnp.zeros(3)))
    gripper_0 = jnp.array(guidance_params.get("__gp_gripper_0__", jnp.zeros(1)))

    # Pre-compute boolean masks for source type selection (avoid repeated comparisons)
    is_finger_l = source_types == 1  # (N,)
    is_finger_r = source_types == 2  # (N,)
    has_fingers = jnp.any(is_finger_l | is_finger_r)

    def _guidance_fn(x_t):
        # x_t: (total_horizon, action_dim) — unbatched.
        T = expected.shape[1]

        # --- Integrate actions to get EE state trajectories ------------------
        if dynamics_fn is not None:
            positions, predicted_ori, predicted_grip = dynamics_fn(x_t[:T], ee_pos_0, ee_ori_0, dt)
        else:
            # Default: Euler integration (cumsum)
            vel = x_t[:, :3]  # (H, 3)
            positions = ee_pos_0[None, :] + jnp.cumsum(vel * dt, axis=0)  # (H, 3)
            positions = positions[:T]  # (T, 3)
            ori_vel = x_t[:T, 3:6]  # (T, 3)
            predicted_ori = ee_ori_0[None, :] + jnp.cumsum(ori_vel * dt, axis=0)  # (T, 3)
            predicted_grip = x_t[:T, 6:7]  # (T, 1) — absolute, not integrated

        # --- Compute finger positions via FK (only when needed) --------------
        # Always compute to keep shapes static for JIT; the cost is minimal
        # when no finger pairs exist (results are unused).
        left_fingers, right_fingers = _jax_finger_positions_vmap(
            positions, predicted_ori, predicted_grip[:, 0]
        )  # each (T, 3)

        # --- Build per-pair source positions based on source_type ------------
        # source_types: 0=EE, 1=finger_l, 2=finger_r, 3=ori, 4=grip
        # For types 3,4 the position is unused but must have valid shape.
        source_pos = jnp.where(
            is_finger_l[None, :, None],
            left_fingers[:, None, :],
            jnp.where(
                is_finger_r[None, :, None],
                right_fingers[:, None, :],
                positions[:, None, :],  # default: EE position
            ),
        )  # (T, N, 3)

        diffs = source_pos - waypoints[None, :, :]  # (T, N, 3)

        # Centroid distances: ||source - waypoint||
        centroid_dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)  # (T, N)

        # Axis distances: dot(waypoint - source, signed_axis)
        axis_dists = -jnp.sum(diffs * axes[None, :, :], axis=-1)  # (T, N)

        # Plane distances: ||in_plane(waypoint - source)||
        proj = jnp.sum(diffs * plane_norms[None, :, :], axis=-1, keepdims=True)  # (T, N, 1)
        in_plane = diffs - proj * plane_norms[None, :, :]  # (T, N, 3)
        plane_dists = jnp.sqrt(jnp.sum(in_plane**2, axis=-1) + 1e-8)  # (T, N)

        # Surface distances: min ||source - surf_pt|| over valid points
        src_exp = source_pos[:, :, None, :]  # (T, N, 1, 3)
        surf_exp = surf_pts[None, :, :, :]  # (1, N, S, 3)
        surf_d2 = jnp.sum((src_exp - surf_exp) ** 2, axis=-1)  # (T, N, S)
        S = surf_pts.shape[1]
        valid_mask = jnp.arange(S)[None, :] < surf_counts[:, None]  # (N, S)
        surf_d2 = jnp.where(valid_mask[None, :, :], surf_d2, jnp.inf)  # (T, N, S)
        surface_dists = jnp.sqrt(jnp.min(surf_d2, axis=-1) + 1e-8)  # (T, N)

        # --- Orientation "distance" = dot(predicted_ori, axis) per pair ------
        ori_dists = jnp.sum(predicted_ori[:, None, :] * axes[None, :, :], axis=-1)  # (T, N)

        # --- Gripper distances -----------------------------------------------
        grip_dists = jnp.broadcast_to(predicted_grip, (T, expected.shape[0]))  # (T, N)

        # --- Select distance per pair based on dist_type ---------------------
        # dist_types: 0=centroid, 1=surface, 2=orientation, 3=gripper
        is_surf = (dist_types == 1)[None, :]  # (1, N)
        is_ori = (dist_types == 2)[None, :]  # (1, N)
        is_grip = (dist_types == 3)[None, :]  # (1, N)
        ha = (has_axis > 0)[None, :]  # (1, N)
        hp = (has_plane > 0)[None, :]  # (1, N)

        # Position-type selection (centroid/axis/plane/surface)
        pos_dists = jnp.where(
            is_surf, surface_dists, jnp.where(ha, axis_dists, jnp.where(hp, plane_dists, centroid_dists))
        )  # (T, N)

        # Final selection across all types
        dists = jnp.where(is_ori, ori_dists, jnp.where(is_grip, grip_dists, pos_dists))  # (T, N)

        errors = dists.T - expected[:, :T]  # (N, T)

        weighted = errors * gf_weights[:, None]  # (N, T)
        return 0.5 * jnp.sum(weighted**2)

    batch_loss_fn = jax.vmap(_guidance_fn)
    batch_grad_fn = jax.vmap(jax.grad(_guidance_fn))
    return batch_loss_fn, batch_grad_fn


def _golden_section_line_search(batch_loss_fn, x_t, direction, max_alpha=1.0, n_iter=20):
    """JIT-compatible golden-section line search for optimal step size.

    Finds alpha in [0, max_alpha] minimizing sum(batch_loss_fn(x_t + alpha * direction)).
    Uses pure JAX control flow (lax.fori_loop) so the entire search is compilable.

    Args:
        batch_loss_fn: (B, T, D) → (B,) per-sample loss.
        x_t: current actions, shape (B, T, D).
        direction: unit-length descent direction, shape (B, T, D).
        max_alpha: initial upper bound for the search bracket.
        n_iter: number of golden-section iterations.

    Returns:
        Scalar alpha (optimal step size).
    """
    gr = (jnp.sqrt(5.0) + 1.0) / 2.0  # golden ratio

    def _total_loss(alpha):
        return jnp.sum(batch_loss_fn(x_t + alpha * direction))

    loss_at_zero = _total_loss(0.0)

    # Bracket expansion: double b up to 5 times while loss decreases
    def _expand_body(carry, _):
        a, b, loss_b = carry
        new_b = b * 2.0
        new_loss = _total_loss(new_b)
        should_expand = new_loss < _total_loss(b)
        b = jnp.where(should_expand, new_b, b)
        loss_b = jnp.where(should_expand, new_loss, loss_b)
        return (a, b, loss_b), None

    init_loss_b = _total_loss(max_alpha)
    (a, b, _), _ = jax.lax.scan(_expand_body, (0.0, max_alpha, init_loss_b), None, length=5)

    # Golden-section narrowing
    def _gs_body(_, carry):
        a, b = carry
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        loss_c = _total_loss(c)
        loss_d = _total_loss(d)
        new_a = jnp.where(loss_c < loss_d, a, c)
        new_b = jnp.where(loss_c < loss_d, d, b)
        return (new_a, new_b)

    a, b = jax.lax.fori_loop(0, n_iter, _gs_body, (a, b))
    alpha = (a + b) / 2.0

    # Only step if it actually reduces loss
    alpha = jnp.where(_total_loss(alpha) < loss_at_zero, alpha, 0.0)
    return alpha


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
                batch_loss_fn, batch_grad_fn = _make_batch_guidance_fns_from_params(
                    guidance_params, dynamics_fn=self._dynamics_fn
                )
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
                # Only compute guidance gradients and alpha on the first action chunk;
                # remaining chunks receive zero guidance.
                x_t_first = x_t[:, : self.action_horizon, :]
                guidance_grad_first = batch_grad_fn(x_t_first + v_t[:, : self.action_horizon, :] * dt)

                # Normalize gradient to unit length per sample (scale-invariant direction)
                grad_norm = jnp.linalg.norm(guidance_grad_first.reshape(batch_size, -1), axis=-1, keepdims=True)[
                    :, :, None
                ]  # (B, 1, 1)
                grad_hat_first = guidance_grad_first / (grad_norm + 1e-8)  # unit ascent direction
                if use_line_search and batch_loss_fn is not None:
                    direction_first = -grad_hat_first  # unit descent direction
                    alpha = _golden_section_line_search(
                        batch_loss_fn,
                        x_t_first,
                        direction_first,
                        max_alpha=1.0,
                    )
                    # Pad with zeros for remaining chunks
                    direction = jnp.concatenate(
                        [direction_first, jnp.zeros_like(x_t[:, self.action_horizon :, :])], axis=1
                    )
                    # Convert position-space step to velocity: dt * v_guidance = alpha * direction
                    v_t = v_t + (alpha / dt) * direction * guidance_factor
                else:
                    # Pad with zeros for remaining chunks
                    grad_hat = jnp.concatenate(
                        [grad_hat_first, jnp.zeros_like(x_t[:, self.action_horizon :, :])], axis=1
                    )
                    # Blend normalized gradient into velocity field.
                    # Since dt < 0, adding +grad_hat to v_t → dt * grad_hat moves in -grad (descent).
                    v_t = v_t + guidance_factor * grad_hat

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
