import os
import sys
import cv2 
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from under_eraser.dist import set_multi_gpus_devices, shard_model
from under_eraser.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                              WanTransformer3DModel)
from under_eraser.models.cache_utils import get_teacache_coefficients
from under_eraser.pipeline.f2f import WanFunInpaintPipeline
from under_eraser.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from under_eraser.utils.lora_utils import merge_lora, unmerge_lora
from under_eraser.utils.utils import (filter_kwargs, get_video_and_mask,get_video_and_mask2,
                                   save_videos_grid)
from under_eraser.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from under_eraser.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from under_eraser.utils.lightx2v_lora_adapter import WanLoraWrapper


GPU_memory_mode     = "sequential_cpu_offload"
ulysses_degree      = 1
ring_degree         = 1
fsdp_dit            = False
fsdp_text_encoder   = True
compile_dit         = False
enable_teacache     = True
teacache_threshold  = 0.10
num_skip_start_steps = 5
teacache_offload    = False
cfg_skip_ratio      = 0
enable_riflex       = False
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
model_name          = "models/Wan2.1-Fun-V1.1-14B-InP"
lightx2v_path       = "weight/lightx2v.safetensors" 
lora_path           = "weight/checkpoint.safetensors"

sampler_name        = "Flow"
shift               = 3 
transformer_path    = None
vae_path            = None
video_length        = 81
fps                 = 16
weight_dtype        = torch.bfloat16
prompt              = ""
negative_prompt     = ""
guidance_scale      = 1.0
seed                = 43
num_inference_steps = 4
lora_weight         = 0.55

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Clip Image Encoder
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype)
clip_image_encoder = clip_image_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanFunInpaintPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

lora_wrapper = WanLoraWrapper(pipeline.transformer)
lora_name = lora_wrapper.load_lora(lightx2v_path)
lora_wrapper.apply_lora(lora_name, 1.0)


if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)


video_dir = "data/video"
mask_dir = "data/mask"
save_path = "result" 



video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]


with torch.no_grad():
    effective_video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    
    for filename in video_files:

        output_video_path = os.path.join(save_path, filename)
        if os.path.exists(output_video_path):
            continue
            
        current_video_path = os.path.join(video_dir, filename)
        current_mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(current_mask_path):
            continue


        vcap = cv2.VideoCapture(current_video_path)
        if vcap.isOpened():
            width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            vcap.release()
            

            if height > width:
                current_sample_size = [720, 480] 
            else:
                current_sample_size = [480, 720]

        else:
            current_sample_size = sample_size

        input_video, input_video_mask, clip_image, clip_neg = get_video_and_mask2(
            input_video_path=current_video_path,
            video_length=effective_video_length,
            sample_size=current_sample_size,
            input_mask_path=current_mask_path
        )
        sample = pipeline(
            prompt, 
            num_frames = effective_video_length,
            negative_prompt = negative_prompt,
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            video      = input_video,
            mask_video   = input_video_mask,
            clip_image = clip_image,
            shift = shift,
        ).videos

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        output_video_path = os.path.join(save_path, filename)
        
        if ulysses_degree * ring_degree > 1:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                save_videos_grid(sample, output_video_path, fps=fps)
        else:
            save_videos_grid(sample, output_video_path, fps=fps)
            
print("Done!")
