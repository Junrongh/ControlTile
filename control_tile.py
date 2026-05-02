from typing import Any

import copy
import warnings

import numpy as np
import cv2

import torch
import torchvision
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar

from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel

from peft import LoraConfig

from parse_args import Args
from utils import get_memory_usage, meshgrid_from_points, fisheye_meshgrid_from_points, swirl_meshgrid_from_points
from model.transform_embedder import TransformEmbedder
from model.transformer import (
    get_transformer,
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
    get_sigmas,
    pack_latents,
    unpack_latents,
    prepare_latent_image_ids,
    calculate_shift,
    retrieve_timesteps
)
from model.encoder import get_encoder_modules
from model.x_embedder import set_x_embedder

warnings.filterwarnings(
    "ignore",
    message="Already found a `peft_config` attribute in the model",
    category=UserWarning,
    module="peft"
)

import os
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

class ControlTileModel(L.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.weight_dtype = args.mix_precision
        self.transformer = FluxTransformer2DModel.from_pretrained(
            args.diffusion_model,
            subfolder="transformer",
            ignore_mismatched_sizes=True,
            torch_dtype=self.weight_dtype,
        )
        self.transformer = get_transformer(self.transformer)
        additional_embedder_channels = [
            ('masked_rgb', 64, 1.0),
            ('mask', 256, 1.0),
            ('irradiance', 64, 1.0),
            ('depth', 64, 1.0), # see FLUX-Depth, use vae to encode depth
        ]
        set_x_embedder(self.transformer, additional_embedder_channels)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.diffusion_model,
            subfolder="scheduler",
        )
        self.noise_scheduler = copy.deepcopy(self.scheduler)

        self._encoder_modules = get_encoder_modules(args.diffusion_model, args.mix_precision, text_encoder=args.text_encoder)
        for v in self._encoder_modules.values():
            v.to(self.device)

        self.vae_scale_factor = 2 ** (len(self._encoder_modules['vae'].config.block_out_channels) - 1) # 8

        if self.args.seed is not None:
            # NOTE: use cpu to keep the random same among devices
            self.generator = torch.Generator('cpu').manual_seed(self.args.seed)
        else:
            self.generator = None

        # LoRA Config
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.1,
            bias="none",
            init_lora_weights="gaussian"
        )
        self.transformer.add_adapter(lora_config, adapter_name='transform_lora')
        self.transformer.set_adapter(['transform_lora'])

        self.validation_step_outputs = []
        self.validation_visual_outputs = []
        self.overfit_visual_outputs = []
        self.trainable_params_list = []
        self.text_embeds = {
            'clip': None,
            't5': None,
        }

    def encode_text_prompt(self, batch_size, clip_text_inputs=None, t5_text_inputs=None):
        if clip_text_inputs is None and t5_text_inputs is None:
            if self.text_embeds['clip'] is None:
                self.text_embeds['clip'] = torch.load(os.path.join(WORKING_DIR, 'data', self.args.clip_text_pt), weights_only=True).to('cpu')
            clip_prompt_embeds = self.text_embeds['clip'].to(self.device)
            if self.text_embeds['t5'] is None:
                self.text_embeds['t5'] = torch.load(os.path.join(WORKING_DIR, 'data', self.args.t5_text_pt), weights_only=True).to('cpu')
            t5_prompt_embeds = self.text_embeds['t5'].to(self.device)
            repeat_fn = lambda x: x.unsqueeze(0).repeat(batch_size, *([1] * x.dim()))
            clip_prompt_embeds, t5_prompt_embeds = map(repeat_fn, (clip_prompt_embeds, t5_prompt_embeds))
        # encoded conditions should be detached from compute graph
        elif self._encoder_modules['clip_encoder'] is not None and self._encoder_modules['t5_encoder'] is not None:
            with torch.no_grad():
                clip_prompt_embeds = self._encoder_modules['clip_encoder'](clip_text_inputs.squeeze(1).to(self.device), output_hidden_states=False)
                clip_prompt_embeds = clip_prompt_embeds.pooler_output
                clip_prompt_embeds = clip_prompt_embeds.to('cpu').detach()
                torch.cuda.empty_cache()

                t5_prompt_embeds = self._encoder_modules['t5_encoder'](t5_text_inputs.squeeze(1).to(self.device), output_hidden_states=False)[0]
                t5_prompt_embeds = t5_prompt_embeds.to('cpu').detach()
                torch.cuda.empty_cache()
        else:
            raise ValueError('clip_text_inputs and t5_text_inputs should be provided or encoder_modules should be provided')
        return clip_prompt_embeds, t5_prompt_embeds

    def image_loss_split(self, model_pred, target, mask=None):
        b, c, h_lat, w_lat = target.shape
        # split loss for each image in batch, for validation to collect validation and overfit images
        loss_mask = torch.nn.functional.interpolate(
            mask.float(), 
            size=(h_lat, w_lat), 
            mode='nearest'
        )
        loss = (loss_mask.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
        num_masked_pixels = loss_mask.view(target.shape[0], -1).sum(dim=1) * c + 1e-6
        loss = loss.view(target.shape[0], -1).sum(dim=1) / num_masked_pixels
        return loss

    def flow_match_loss(self, model_pred, target, mask, weighting=None):
        _, c, h_lat, w_lat = target.shape
        # split loss for each image in batch, for validation to collect validation and overfit images
        if weighting is None:
            weighting = torch.ones_like(target)
        loss_mask = torch.nn.functional.interpolate(
            mask.float(), 
            size=(h_lat, w_lat), 
            mode='nearest'
        )
        loss_mask = (loss_mask > 0.5).float()
        loss_mask = loss_mask * 0.7 + 0.3
        loss = (weighting.float() * loss_mask.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
        num_masked_pixels = loss_mask.sum() * c + 1e-6
        loss = loss.sum() / num_masked_pixels
        return loss

    def _vae_encode(self, x):
        x = self._encoder_modules['vae'].encode(x.to(self.device)).latent_dist.sample(self.generator)
        x = x * self._encoder_modules['vae'].config.scaling_factor
        return x

    def _vae_decode(self, x):
        x = x / self._encoder_modules['vae'].config.scaling_factor
        x = self._encoder_modules['vae'].decode(x, return_dict=False)[0]
        return x

    def _common_process(self, batch, is_inference=False):

        target = batch['gt'].to(self.weight_dtype)
        masked_rgb = batch['masked_rgb'].to(self.weight_dtype)
        mask = batch['mask'].to(self.weight_dtype)
        mask = (mask > 0.1) + 0.0
        geometry = batch['geometry'].to(self.weight_dtype)
        irradiance = batch['irradiance'].to(self.weight_dtype)
        mtl = batch['mtl'].to(self.weight_dtype)
        batch_size, C, H, W = target.shape

        visualize_params = {'batch_size': batch_size}
        visualize_params['gt'] = (target / 2 + 0.5).clamp(0, 1)

        coords = batch['coords'].to(self.weight_dtype)

        target = self._vae_encode(target)
        # target need to add noise level in training step
        batch_size, _, height, width = target.shape

        # channel concat condition is packed here        
        irradiance = self._vae_encode(irradiance)
        irradiance = pack_latents(irradiance)
        geometry = self._vae_encode(geometry)
        geometry = pack_latents(geometry)

        clip_text_embeds, t5_text_embeds = self.encode_text_prompt(batch_size)

        text_ids = torch.zeros(t5_text_embeds.shape[1], 3)

        image_ids = prepare_latent_image_ids(height // 2, width // 2) # [1024, 3]

        if is_inference:
            visualize_params['condition'] = (mtl / 2 + 0.5).clamp(0, 1)
            points = coords.float().cpu().numpy()
            mask_pt_list = []
            for b in range(batch_size):
                mask_np = mask[b].to(torch.float).cpu().numpy().transpose(1, 2, 0)
                mask_np = np.concatenate([mask_np, mask_np, mask_np], axis=-1)
                cv2.circle(mask_np, (int(points[b][0][1] * W), int(points[b][0][0] * H)), 5, (0, 0, 1), -1)
                cv2.circle(mask_np, (int(points[b][1][1] * W), int(points[b][1][0] * H)), 5, (0, 1, 0), -1)
                cv2.circle(mask_np, (int(points[b][2][1] * W), int(points[b][2][0] * H)), 5, (1, 0, 0), -1)
                cv2.circle(mask_np, (int(points[b][3][1] * W), int(points[b][3][0] * H)), 5, (1, 0, 1), -1)
                mask_pt = torch.tensor(mask_np).permute(2, 0, 1).to(self.device) # 3, h, w
                mask_pt_list.append(mask_pt.unsqueeze(0)) # 1, 3, h, w
            visualize_params['mask'] = torch.cat(mask_pt_list, dim=0) # b, 3, h, w
        masked_rgb = self._vae_encode(masked_rgb)
        masked_rgb = pack_latents(masked_rgb)
    
        mtl = self._vae_encode(mtl)
        _, _, c_height, c_width = mtl.shape
        mtl = pack_latents(mtl)
        mtl_ids = meshgrid_from_points(coords, height // 2, width // 2, c_height // 2, c_width // 2, additional_value=0.0) # [B, 1024, 3]
        condition = mtl
        condition_ids = mtl_ids

        # 5.resize mask to latents shape we we concatenate the mask to the latents
        mask_pt = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask_pt = mask_pt.view(
            batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask_pt = mask_pt.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask_pt = mask_pt.reshape(
            batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width
        mask_pt = pack_latents(mask_pt)
        if self.args.attn_mask:
            attn_mask = mask_pt.sum(axis=2) > 1.0
        else:
            attn_mask = None

        channel_concat_condition = torch.cat([masked_rgb, mask_pt, irradiance, geometry], dim=2)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], 30.0)
            guidance = guidance.expand(batch_size).to(self.device)
        else:
            guidance = None

        return (
            target,
            condition,
            channel_concat_condition,
            image_ids,
            condition_ids,
            clip_text_embeds,
            t5_text_embeds,
            text_ids,
            guidance,
            mask,
            attn_mask,
            visualize_params,
        )

    def training_step(self, batch, batch_idx):
        target, condition_latent, channel_concat_condition, image_ids, condition_ids, clip_text_embeds, t5_text_embeds, text_ids, guidance, mask, attn_mask, visualize_params = self._common_process(batch)

        batch_size, _, height, width = target.shape
        noise = torch.randn_like(target)
        loss_target = noise - target

        u = compute_density_for_timestep_sampling(
            weighting_scheme=None,
            batch_size=batch_size,
            logit_mean=None,
            logit_std=None,
            mode_scale=None,
        )

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=self.device)
        sigmas = get_sigmas(scheduler=self.noise_scheduler, timesteps=timesteps, n_dim=target.ndim, device=self.device)

        latents = (1.0 - sigmas) * target + sigmas * noise
        packed_latents = pack_latents(latents)

        latent_model_input = torch.cat([packed_latents, channel_concat_condition], dim=2) # channel concat

        # different from FLUX-Kontext:
        # condition latents should be passed through original x_embedder
        # since it has no channel concat condition
        model_pred = self.transformer(
            hidden_states=latent_model_input,
            condition_latents=condition_latent,
            encoder_hidden_states=t5_text_embeds.to(self.device),
            pooled_projections=clip_text_embeds.to(self.device),
            img_ids=image_ids,
            condition_ids=condition_ids,
            txt_ids=text_ids,
            timestep=timesteps / 1000,
            guidance=guidance,
            attn_mask=attn_mask,
            joint_attention_kwargs=None,
        )[0]

        model_pred = model_pred[:, :packed_latents.size(1)]
        pred = unpack_latents(model_pred, height, width)
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)

        flow_match_loss = self.flow_match_loss(pred, loss_target, mask, weighting)

        self.log(f"fm_loss", flow_match_loss.mean(), prog_bar=True, on_step=True, sync_dist=False)
        loss = flow_match_loss.mean()
        tensorboard = self.logger.experiment
        tensorboard.add_scalar("train/loss", loss, global_step=self.global_step * self.args.batch_size)
        return { "loss": loss }

    def on_train_start(self) -> None:
        for encoder in self._encoder_modules.values():
            if encoder.device != self.device:
                encoder.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # The value outputs["loss"] here will be the normalized value
        # w.r.t accumulate_grad_batches of the loss returned from training_step.
        # self.log("train/loss-batch", outputs["loss"], on_step=True, sync_dist=True)
        return

    def on_after_backward(self) -> None:
        # get_memory_usage('train', [self.transformer], self.optimizer)
        return

    def _inference(self, batch, use_progress_bar=False):
        target, condition_latent, channel_concat_condition, image_ids, condition_ids, clip_text_embeds, t5_text_embeds, text_ids, guidance, mask, attn_mask, visualize_params = self._common_process(batch, is_inference=True)

        batch_size, _, height, width = target.shape
        sigmas = np.linspace(1.0, 1 / self.args.inference_timesteps, self.args.inference_timesteps)
        image_seq_len = (height // 2) * (width // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            self.args.inference_timesteps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        noise = torch.randn_like(target)
        latents = pack_latents(noise)

        if use_progress_bar:
            progress_bar = self._get_progress_bar()
            if progress_bar and hasattr(self, 'inference_task'):
                use_progress_bar = True
                progress_bar.progress.update(self.inference_task, completed=0)
            else:
                use_progress_bar = False

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            latent_model_input = torch.cat([latents, channel_concat_condition], dim=2) # channel concat
            model_pred = self.transformer(
                hidden_states=latent_model_input,
                condition_latents=condition_latent,
                encoder_hidden_states=t5_text_embeds.to(self.device),
                pooled_projections=clip_text_embeds.to(self.device),
                img_ids=image_ids,
                condition_ids=condition_ids,
                txt_ids=text_ids,
                timestep=t_batch / 1000,
                guidance=guidance,
                attn_mask=attn_mask,
                joint_attention_kwargs=None,
            )[0]

            model_pred = model_pred[:, : latents.size(1)]
            latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]
            if use_progress_bar:
                progress_bar.progress.update(self.inference_task, advance=1)
                progress_bar.refresh()

        model_pred = unpack_latents(latents, height, width)
        image = self._vae_decode(model_pred)
        latent_loss = self.flow_match_loss(model_pred, target, mask)
        image_loss = self.image_loss_split(image, batch['gt'], mask)
        return image, latent_loss, image_loss, visualize_params

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images, _, image_losses, visualize_params = self._inference(batch)
            tags = batch['tag']
            batch_size = visualize_params['batch_size']
            for idx in range(batch_size):
                tag = tags[idx]
                image = images[idx].unsqueeze(0)
                image_loss = image_losses[idx]
                gt = visualize_params['gt'][idx].unsqueeze(0)
                condition = visualize_params['condition'][idx].unsqueeze(0)
                mask = visualize_params['mask'][idx].unsqueeze(0)
                resize = torchvision.transforms.Resize(self.args.resolution)
                condition = resize(condition)
                mask = resize(mask)
                if 'val' in tag:
                    self.validation_step_outputs.append(image_loss)
                if '_visualize' in tag:
                    inference = (image / 2 + 0.5).clamp(0, 1)
                    img_grid = torchvision.utils.make_grid(torch.cat([
                        gt,
                        inference,
                        mask,
                        condition,
                    ], dim=0), nrow=1)
                    if 'overfit' in tag:
                        self.overfit_visual_outputs.append(img_grid)
                    else: # 'val' in tag
                        self.validation_visual_outputs.append(img_grid)

    def on_validation_start(self) -> None:
        for encoder in self._encoder_modules.values():
            if encoder.device != self.device:
                encoder.to(self.device)

    def on_validation_epoch_end(self) -> None:
        # get_memory_usage('val', [self.transformer], self.optimizer)

        if len(self.validation_step_outputs) > 0:
            outputs = torch.stack(self.validation_step_outputs)
            mean = torch.mean(self.all_gather(outputs))
            self.validation_step_outputs.clear()
            self.log("val_loss", mean, sync_dist=True, rank_zero_only=True)

        if len(self.validation_visual_outputs) > 0:
            val_visual = torch.stack(self.validation_visual_outputs)
            val_visual = self.all_gather(val_visual)
            val_visual = val_visual.reshape(-1, *val_visual.shape[2:])
            val_visual = [val_visual[i] for i in range(val_visual.shape[0])]
            img_grid = torch.cat(val_visual, dim=-1)
            self.logger.experiment.add_image('Validation', img_grid, self.current_epoch)
            self.validation_visual_outputs.clear()

        if len(self.overfit_visual_outputs) > 0:
            overfit_visual = torch.stack(self.overfit_visual_outputs)
            overfit_visual = self.all_gather(overfit_visual)
            overfit_visual = overfit_visual.reshape(-1, *overfit_visual.shape[2:])
            overfit_visual = [overfit_visual[i] for i in range(overfit_visual.shape[0])]
            img_grid = torch.cat(overfit_visual, dim=-1)
            self.logger.experiment.add_image('Overfit', img_grid, self.current_epoch)
            self.overfit_visual_outputs.clear()

        torch.cuda.empty_cache()

    def _get_progress_bar(self):
        # for interactive task inference, use RichProgressBar
        if not hasattr(self.trainer, 'progress_bar_callback'):
            return None
            
        for callback in self.trainer.callbacks:
            if isinstance(callback, RichProgressBar):
                return callback
        return None

    def on_test_start(self) -> None:
        progress_bar = self._get_progress_bar()
        if progress_bar:
            self.inference_task = progress_bar.progress.add_task(
                "[cyan]Inference...",
                total=self.args.inference_timesteps
            )
        for encoder in self._encoder_modules.values():
            if encoder.device != self.device:
                encoder.to(self.device)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            images, _, _, visualize_params = self._inference(batch, use_progress_bar=True)
            tags = batch['tag']
            batch_size = visualize_params['batch_size']
            for idx in range(batch_size):
                inference = images[idx] * 0.5 + 0.5
                gt = visualize_params['gt'][idx]
                mask = visualize_params['mask'][idx]
                condition = visualize_params['condition'][idx]
                concat_list = []
                for i in [gt, inference, mask, condition]:
                    i = i.to(torch.float).cpu().numpy() # c, h, w
                    if i.ndim == 2:
                        i = i[np.newaxis, ...]
                        i = np.concatenate([i, i, i], axis=0)
                    i = i.transpose(1, 2, 0)
                    i = cv2.resize(i, (self.args.resolution))
                    concat_list.append(i)

                concat = np.concatenate(concat_list, axis=1)
                concat = concat[..., ::-1]
                tag = tags[idx]
                name = batch['name'][idx]
                cv2.imwrite(f'{self.args.img_output_folder}/{name}_{tag}.png', np.clip(concat * 255, 0, 255).astype(np.uint8))
                print(f'{self.args.img_output_folder}/{name}_{tag}.png')

    def configure_optimizers(self):
        self.transformer.train()
        trainable_params = []

        # set conv_in layers to trainable
        self.transformer.x_embedder.requires_grad_(True)
        for n, p in self.transformer.x_embedder.named_parameters():
            trainable_params.append(p)
            self.trainable_params_list.append(f'transformer.x_embedder.{n}')

        lora_layers = filter( lambda p: f'transform_lora' in p[0], self.transformer.named_parameters() )
        for k, v in lora_layers:
            v.requires_grad_(True)
            trainable_params.append(v)
            self.trainable_params_list.append(f'transformer.{k}')

        NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]
        for name, param in self.transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad_(True)
                trainable_params.append(param)
                self.trainable_params_list.append(f'transformer.{name}')

        with open('notes/tpdm/state_dict.keys', 'w') as f:
            for k, v in self.transformer.named_parameters():
                f.write(k + '\n')

        optimizer = torch.optim.AdamW(trainable_params, lr=self.args.learning_rate)
        self.optimizer = optimizer
        # we dont need scheduler for AdamW
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        with open('notes/tpdm/trainable_params.txt', 'w') as f:
            for param in self.trainable_params_list:
                f.write(param + '\n')
        return [optimizer] # [scheduler]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        save_keys = self.trainable_params_list

        keys = list(checkpoint['state_dict'].keys())
        for k in keys:
            if k not in save_keys:
                del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        load_keys = list(checkpoint['state_dict'].keys())
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for k in keys:
            if k not in load_keys:
                checkpoint['state_dict'][k] = state_dict[k]

