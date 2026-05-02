from typing import Optional, Union, Tuple, List, Any, Dict
from types import MethodType
import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.attention import Attention
from diffusers.utils import USE_PEFT_BACKEND, unscale_lora_layers, scale_lora_layers
from diffusers.models.embeddings import apply_rotary_emb
from model.transform_embedder import FluxPosEmbedBatch

from contextlib import contextmanager

from peft.tuners.tuners_utils import BaseTunerLayer

@contextmanager
def specify_lora(lora_modules: List[BaseTunerLayer], specified_lora=None):
    # Filter valid lora modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    # Save original scales
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    # Enter context: adjust scaling
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    try:
        yield
    finally:
        # Exit context: restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]

def get_transformer(transformer: FluxTransformer2DModel) -> FluxTransformer2DModel:
    transformer.pos_embed = FluxPosEmbedBatch(theta=10000, axes_dim=(16, 56, 56))
    transformer.forward = MethodType(transformer_forward, transformer)
    for block in transformer.transformer_blocks:
        block.forward = MethodType(transformer_block_forward, block)
    for block in transformer.single_transformer_blocks:
        block.forward = MethodType(single_transformer_block_forward, block)

    transformer.enable_gradient_checkpointing()
    # transformer.gradient_checkpointing = True
    # transformer._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    return transformer

def transformer_forward(
    self: FluxTransformer2DModel,
    hidden_states: torch.Tensor,
    condition_latents: torch.Tensor = None,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    condition_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    attn_mask: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor]:

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)

    batch_size = hidden_states.shape[0]
    hidden_states = self.x_embedder(hidden_states)
    if condition_latents is not None:
        # in FLUX-Fill, condition latents needs mask and masked rgb as the additional channels
        # therefore, condition_latents should concat ones and itself
        b, s, c = condition_latents.shape
        condition_latents = torch.cat([
            condition_latents,
            condition_latents, # fake masked rgb
            torch.ones(b, s, c * 4).to(condition_latents.device), # fake mask
            torch.ones(b, s, c).to(condition_latents.device), # fake irradiance
            torch.ones(b, s, c).to(condition_latents.device), # fake geometry
        ], dim=2) # [b, 1024, 64] -> [b, 1024, 512]
        condition_latents = self.x_embedder(condition_latents)
        hidden_states = torch.cat([hidden_states, condition_latents], dim=1) # 2 x [b, 1024, 3072] -> [b, 2048, 3072]

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    txt_ids = txt_ids.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.float32)
    img_ids = img_ids.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.float32)
    ids = torch.cat([txt_ids, img_ids, condition_ids], dim=1) # [B, seq_len, 3], seq_len = 512+1024+1024
    image_rotary_emb = self.pos_embed(ids) # ([B, seq_len, 128], [B, seq_len, 128])

    for _, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attn_mask,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, # B, 2048, 3072
                encoder_hidden_states=encoder_hidden_states, # B, 512, 3072
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attn_mask=attn_mask,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    for _, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attn_mask,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attn_mask=attn_mask,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    return (output,)

def create_attn_mask(mask_packed, text_seq_len, target_seq_len, condition_seq_len, dtype=torch.bfloat16):
    b, _ = mask_packed.shape
    device = mask_packed.device

    total_len = text_seq_len + target_seq_len + condition_seq_len
    attn_bias = torch.zeros((b, total_len, total_len), dtype=dtype, device=device)

    is_bg_region = ~mask_packed
    zeros_text = torch.zeros(b, text_seq_len, dtype=torch.bool, device=device)
    zeros_cond = torch.zeros(b, condition_seq_len, dtype=torch.bool, device=device)
    query_is_bg = torch.cat([zeros_text, is_bg_region, zeros_cond], dim=1)
    
    zeros_text_target = torch.zeros((b, text_seq_len + target_seq_len), dtype=torch.bool, device=device)
    ones_cond = torch.ones((b, condition_seq_len), dtype=torch.bool, device=device)

    key_is_cond = torch.cat([zeros_text_target, ones_cond], dim=1) # [B, Total]
    mask_to_block = query_is_bg.unsqueeze(2) & key_is_cond.unsqueeze(1)
    
    min_val = torch.finfo(dtype).min
    attn_bias.masked_fill_(mask_to_block, min_val)

    cond_start_idx = text_seq_len + target_seq_len
    attn_bias[:, cond_start_idx:, :cond_start_idx] = min_val

    return attn_bias

def transformer_block_forward(
    self: FluxTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[List[torch.Tensor]] = None,
    attn_mask: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    if encoder_hidden_states is not None:
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

    if attn_mask is not None:
        img_seq_len = hidden_states.shape[1]
        target_seq_len = attn_mask.shape[1]
        condition_seq_len = img_seq_len - target_seq_len
        text_seq_len = encoder_hidden_states.shape[1]
        attention_mask = create_attn_mask(attn_mask, text_seq_len, target_seq_len, condition_seq_len, dtype=hidden_states.dtype)
    else:
        attention_mask = None

    joint_attention_kwargs = joint_attention_kwargs or {}
    # Attention.
    attn_processor = attn_processor_forward
    attention_outputs = attn_processor(
        self.attn,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        attn_mask=attention_mask,
        **joint_attention_kwargs,
    )

    attn_output, context_attn_output = attention_outputs

    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output
    hidden_states = hidden_states + ff_output

    # Process attention outputs for the `encoder_hidden_states`.
    if encoder_hidden_states is not None:
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states

def single_transformer_block_forward(
    self: FluxSingleTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[List[torch.Tensor]] = None,
    attn_mask: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    text_seq_len = encoder_hidden_states.shape[1]
    img_seq_len = hidden_states.shape[1]

    if attn_mask is not None:
        target_seq_len = attn_mask.shape[1]
        condition_seq_len = img_seq_len - target_seq_len
        attention_mask = create_attn_mask(attn_mask, text_seq_len, target_seq_len, condition_seq_len, dtype=hidden_states.dtype)
    else:
        attention_mask = None

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}

    attn_processor = attn_processor_forward
    attn_output = attn_processor(
        self.attn,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        attn_mask=attention_mask,
        **joint_attention_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = torch.split(hidden_states, [text_seq_len, img_seq_len], dim=1)
    return encoder_hidden_states, hidden_states

def attn_processor_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attn_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[List[torch.Tensor]] = None
):
    batch_size, _, _ = hidden_states.shape
    reshape_fn = lambda x: x.view(batch_size, -1, attn.heads, x.shape[-1] // attn.heads).transpose(1, 2)

    if encoder_hidden_states is not None:
        query_context = attn.add_q_proj(encoder_hidden_states)
        key_context = attn.add_k_proj(encoder_hidden_states)
        value_context = attn.add_v_proj(encoder_hidden_states)

        query_context, key_context, value_context = map(reshape_fn, (query_context, key_context, value_context))
        query_context, key_context = attn.norm_added_q(query_context), attn.norm_added_k(key_context)

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query, key, value = map(reshape_fn, (query, key, value))
    query, key = attn.norm_q(query), attn.norm_k(key)

    if encoder_hidden_states is not None:
        querys = torch.cat([query_context, query], dim=2)
        keys = torch.cat([key_context, key], dim=2)
        values = torch.cat([value_context, value], dim=2)
    else:
        querys = query
        keys = key
        values = value

    querys = apply_rotary_emb_batch(querys, image_rotary_emb)
    keys = apply_rotary_emb_batch(keys, image_rotary_emb)
    if attn_mask is not None:
        bias = attn_mask.unsqueeze(1)
    else:
        bias = None

    attn_output = F.scaled_dot_product_attention(
        querys,
        keys,
        values,
        attn_mask=bias,
        dropout_p=0.0,
        is_causal=False
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, hidden_states.shape[-1])
    attn_output = attn_output.to(query.dtype)

    if encoder_hidden_states is not None:
        context_attn_output, attn_output = (
            attn_output[:, :encoder_hidden_states.shape[1]],
            attn_output[:, encoder_hidden_states.shape[1]:]
        )

        context_attn_output = attn.to_add_out(context_attn_output)
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)
        return attn_output, context_attn_output
    else:
        return attn_output

def compute_density_for_timestep_sampling(
    weighting_scheme: str, # ["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"], "none" weighting scheme for uniform sampling and uniform loss
    batch_size: int,
    logit_mean: float = None, # mean to use when using the `'logit_normal'` weighting scheme.
    logit_std: float = None, # std to use when using the `'logit_normal'` weighting scheme.
    mode_scale: float = None, # Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def get_sigmas(scheduler, timesteps, n_dim=4, device=None):
    sigmas = scheduler.sigmas.to(device)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def pack_latents(latents: torch.Tensor, scale_factor: int=2):
    batch_size, num_channels_latents, height, width = latents.shape
    scaled_height, scaled_width = height // scale_factor, width // scale_factor
    latents = latents.view(batch_size, num_channels_latents, scaled_height, scale_factor, scaled_width, scale_factor)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, scaled_height * scaled_width, num_channels_latents * scale_factor * scale_factor)

    return latents

def prepare_latent_image_ids(height, width):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids

def unpack_latents(latents, height, width):
    batch_size, _, channels = latents.shape

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def apply_rotary_emb_batch(
    pes: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [S, D]
    cos, sin = cos.to(pes.device), sin.to(pes.device)

    outs = []
    for x, c, s in zip(pes, cos, sin):
        x = x.unsqueeze(0)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * c + x_rotated.float() * s).to(x.dtype)
        outs.append(out)

    out = torch.cat(outs, dim=0)

    return out
