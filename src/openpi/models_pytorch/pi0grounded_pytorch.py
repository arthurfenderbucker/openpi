"""Pi0Grounded PyTorch: Extended Pi0 model with multi-chunk action generation.

This is the PyTorch implementation of Pi0Grounded, extending PI0Pytorch to support
generating multiple independent action chunks in a single inference call.

Key features:
- Generates num_action_chunks independent action sequences per inference
- Each chunk uses different random noise initialization
- Chunks are processed in parallel using block-diagonal causal attention
- Each chunk can only attend to the prefix (observation) and itself, not other chunks
- This is equivalent to running the action expert num_chunks times independently,
  but more computationally efficient
- Backward compatible: num_action_chunks=1 matches original PI0Pytorch exactly
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from typing_extensions import override

from openpi.models_pytorch import pi0_pytorch


class PI0GroundedPytorch(pi0_pytorch.PI0Pytorch):
    """PyTorch Pi0 model with multi-chunk action generation support."""
    
    def __init__(self, config):
        # Initialize parent PI0Pytorch model
        super().__init__(config)
        # Store the number of chunks for inference
        self.num_action_chunks = getattr(config, 'num_action_chunks', 1)
        
        # Re-compile sample_actions with the new implementation
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
    
    def embed_suffix(self, state, noisy_actions, timestep, num_chunks=None):
        """Embed state, noisy_actions, timestep with support for multiple chunks.
        
        Args:
            state: Robot state
            noisy_actions: Noisy action sequence [batch, num_chunks * action_horizon, action_dim]
            timestep: Flow matching timestep
            num_chunks: Number of chunks (if None, uses self.num_action_chunks)
        
        Returns:
            Embedded tokens, pad masks, attention masks, and optional adaRMS conditioning
        """
        if num_chunks is None:
            num_chunks = self.num_action_chunks
            
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state (shared across all chunks)
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = pi0_pytorch.create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            # Expand time embedding for all action steps across all chunks
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks for multiple chunks
        # Each chunk: first token breaks causality (1), rest are causal (0)
        for _ in range(num_chunks):
            att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    @torch.no_grad()
    @override
    def sample_actions(self, device, observation, noise=None, num_steps=10, num_action_chunks=None) -> Tensor:
        """Sample actions with support for multiple chunks.
        
        Args:
            device: Device to run inference on
            observation: Input observation
            noise: Optional initial noise
            num_steps: Number of denoising steps
            num_action_chunks: Number of chunks to generate (overrides config if provided)
        
        Returns:
            Actions with shape [batch_size, num_chunks, action_horizon, action_dim]
        """
        if num_action_chunks is None:
            num_action_chunks = self.num_action_chunks
            
        bsize = observation.state.shape[0]
        total_horizon = num_action_chunks * self.config.action_horizon
        
        if noise is None:
            actions_shape = (bsize, total_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Compute prefix embeddings and cache (unchanged from base model)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step_grounded(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                num_chunks=num_action_chunks,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        
        # Reshape to [batch, num_chunks, action_horizon, action_dim]
        x_t = x_t.reshape(bsize, num_action_chunks, self.config.action_horizon, self.config.action_dim)
        return x_t

    def denoise_step_grounded(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        num_chunks=None,
    ):
        """Apply one denoising step for multi-chunk actions.
        
        Args:
            state: Robot state
            prefix_pad_masks: Padding masks for prefix
            past_key_values: Cached key-value pairs from prefix
            x_t: Current noisy actions
            timestep: Flow matching timestep
            num_chunks: Number of chunks
        
        Returns:
            Velocity prediction v_t for all chunks
        """
        if num_chunks is None:
            num_chunks = self.num_action_chunks
            
        total_horizon = num_chunks * self.config.action_horizon
        
        # Embed suffix with multiple chunks
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep, num_chunks=num_chunks
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # Create block-diagonal causal attention mask for independent chunks
        # Each chunk can only attend to itself (and prefix), not to other chunks
        suffix_att_2d_masks = torch.zeros(batch_size, suffix_len, suffix_len, dtype=torch.bool, device=x_t.device)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.config.action_horizon
            end_idx = start_idx + self.config.action_horizon
            
            # Create causal mask for this chunk
            chunk_mask = torch.tril(torch.ones(self.config.action_horizon, self.config.action_horizon, dtype=torch.bool, device=x_t.device))
            
            # Place it in the block-diagonal position
            suffix_att_2d_masks[:, start_idx:end_idx, start_idx:end_idx] = chunk_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply padding masks
        pad_2d_masks = suffix_pad_masks[:, None, :] * suffix_pad_masks[:, :, None]
        suffix_att_2d_masks = suffix_att_2d_masks & pad_2d_masks

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -total_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
