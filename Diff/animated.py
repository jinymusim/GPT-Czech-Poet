import torch
import random
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("strangeman3107/animov-0.1.1", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()


i= 0
possible_prompts = ["1girl, walking, boobs, colorful, sweat",
                    "1girl, bikini, colorful",]

while i < 10:  
    prompt = random.choice(possible_prompts)
    
    video_frames = pipe(prompt, num_inference_steps=200).frames
    video_path = export_to_video(video_frames,  prompt + f'{i}.mp4')
    
    i+=1