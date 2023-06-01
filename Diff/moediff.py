from diffusers import StableDiffusionPipeline
import random
import torch
#from espnet2.bin.tts_inference import Text2Speech



# "Ojimi/anime-kawai-diffusion"
# "dreamlike-art/dreamlike-anime-1.0"
# "FredZhang7/anime-anything-promptgen-v2"
pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-anime-1.0", safety_checker=None, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
i= 0

#model = Text2Speech.from_pretrained("mio/amadeus")
#speech, *_ = model("text to generate speech from")

# "1girl, colorful, underwear",  
# "1girl, ass, lingerie, colorful",
# "1girl, boobs, underwear, underboob, colorful",
# "1girl, boobs, underwear, underboob, big boobs, colorful",
# "1women, mature women, wide hips, big hips, color clothes"
# "women, wide hips, big hips, lingerie, garter belt, dim lighting, ass"
# "girl, boobs, underwear, wide hips, some colors, high quality, more realistic, highest quality, masterpiece",
# "girl, wide hips, underwear, boobs, colorful, high quality, more realistic, dim light, artificial light, highest quality, masterpiece
# "1girl, weight gain, lingerie, colorful",
# "1girl, weight gain, chubby, underwear, colorful",
# "2girls, ass, lingerie, colorful",
# "1girl, plumb, fatten, weight gain, chubby, lingerie",

# More Fantasy Like
# "1girl, boobs, boob armor, six pack, battle scars, sweat, some colors, more colors, high quality, highest quality, masterpiece"
# "1girl, boobs, boob armor, six pack, blood, sweat, some colors, more colors, high quality, highest quality, masterpiece"
possible_prompts = ["1woman, garter belt, dim lighting, dim light, realistic, highest quality, masterpiece",
                    "1woman, underwear, underboob, colorful, realistic, highest quality, masterpiece",
                    "1woman, boobs, boob armor, six pack, blood, sweat, some colors, more colors, high quality, highest quality, masterpiece",
                    "1woman, six pack, blood, colorful, high quality, highest quality, masterpiece",]

while i< 25:  
    prompt = random.choice(possible_prompts)
    
    image = pipe(prompt, negative_prompt="""simple background, retro style, low quality, lowest quality, 
                 1980s, 1990s, 2000s, bad anatomy, bad proportions, lowres, username, artist name, 
                 error, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry""",
                 num_inference_steps=999).images[0]
    image.save(prompt + f"{i}.png")
    
    i+=1