
import torch
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel

prompt = f"Seamlessly map the reference texture into the black masked region of the scene. The texture must strictly retain its original structural details and high-frequency patterns without distortion. Ensure the mapped texture follows the scene's geometric perspective and physically accurate lighting, including proper shading and ambient occlusion to match the global illumination. The surrounding unmasked environment must remain pixel-perfect and unchanged"
prompt = [prompt]

diffusion_model_path = "/home/jrhuang8/scratch/model/FLUX-Kontext"

clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
    diffusion_model_path,
    subfolder="tokenizer"
)
t5_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(
    diffusion_model_path,
    subfolder="tokenizer_2"
)

clip_text_inputs = clip_tokenizer(
    prompt,
    padding="max_length",
    max_length=clip_tokenizer.model_max_length,
    truncation=True,
    return_overflowing_tokens=False,
    return_length=False,
    return_tensors="pt"
).input_ids

t5_text_inputs = t5_tokenizer(
    prompt,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_length=False,
    return_overflowing_tokens=False,
    return_tensors="pt",
).input_ids

calc_device = 'cuda:0'
save_device = 'cpu'

clip_encoder = CLIPTextModel.from_pretrained(
    diffusion_model_path,
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
).to(calc_device)
clip_encoder.eval()
clip_encoder.requires_grad_(False)

t5_encoder = T5EncoderModel.from_pretrained(
    diffusion_model_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16,
).to(calc_device)
t5_encoder.eval()
t5_encoder.requires_grad_(False)

with torch.no_grad():
    clip_prompt_embeds = clip_encoder(clip_text_inputs.squeeze(1).to(calc_device), output_hidden_states=False)
    clip_prompt_embeds = clip_prompt_embeds.pooler_output
    clip_prompt_embeds = clip_prompt_embeds.to(save_device).detach()
    torch.cuda.empty_cache()

    t5_prompt_embeds = t5_encoder(t5_text_inputs.squeeze(1).to(calc_device), output_hidden_states=False)[0]
    t5_prompt_embeds = t5_prompt_embeds.to(save_device).detach()
    torch.cuda.empty_cache()

clip_prompt_embeds = clip_prompt_embeds.squeeze(0)
t5_prompt_embeds = t5_prompt_embeds.squeeze(0)

print(clip_prompt_embeds.shape) # should be [768]
print(t5_prompt_embeds.shape) # should be [512, 4096]

torch.save(clip_prompt_embeds, 'clip_text_embeds_preview.pt')
torch.save(t5_prompt_embeds, 't5_text_embeds_preview.pt')
