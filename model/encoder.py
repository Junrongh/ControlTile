from diffusers import AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel

def get_encoder_modules(diffusion_model, mix_precision, text_encoder=True):
    vae = AutoencoderKL.from_pretrained(
        diffusion_model,
        subfolder="vae",
        torch_dtype=mix_precision,
    ).to('cpu')
    vae.eval()
    vae.requires_grad_(False)

    if text_encoder:
        clip_encoder = CLIPTextModel.from_pretrained(
            diffusion_model,
            subfolder="text_encoder",
            torch_dtype=mix_precision,
        ).to('cpu')
        clip_encoder.eval()
        clip_encoder.requires_grad_(False)

        t5_encoder = T5EncoderModel.from_pretrained(
            diffusion_model,
            subfolder="text_encoder_2",
            torch_dtype=mix_precision,
        ).to('cpu')
        t5_encoder.eval()
        t5_encoder.requires_grad_(False)
        return {
            'vae': vae,
            'clip_encoder': clip_encoder,
            't5_encoder': t5_encoder,
        }
    else:
        return {
            'vae': vae,
        }
