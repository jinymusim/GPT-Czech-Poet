from diffusers import StableDiffusionPipeline
import random
import torch
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

# "Ojimi/anime-kawai-diffusion"
# "dreamlike-art/dreamlike-anime-1.0"
# "FredZhang7/anime-anything-promptgen-v2"
pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-anime-1.0", safety_checker=None, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# "mio/tokiwa_midori"
text2speech = Text2Speech.from_pretrained("mio/Artoria")
i= 0

# "1girl, colorful, underwear",  
# "1girl, ass, lingerie, colorful",
# "1girl, boobs, underwear, underboob, colorful" : "優しくしてください",
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
possible_prompts = {"1girl, boobs, underwear, underboob, colorful" : ["優しくしてください。","行かないでください。", 
                                                                      '私を一人にしないでください。', 'あなたのこと待ってたのに。', 
                                                                      'あなたに会えなくてとても寂しい。','本当にあなたがいなくて寂しいです、もう私から離れないでください。']}

while i< 25:  
    prompt = random.choice(list(possible_prompts.keys()))
    
    image = pipe(prompt, negative_prompt="""simple background, retro style, low quality, lowest quality, 
                 1980s, 1990s, 2000s, bad anatomy, bad proportions, lowres, username, artist name, 
                 error, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry""",
                 num_inference_steps=100).images[0]   
    image.save(prompt + f"{i}.png")
    
    speech = text2speech(random.choice(possible_prompts[prompt]))["wav"]
    sf.write(prompt + f"{i}.wav", speech.numpy(), text2speech.fs, "PCM_16")

    i+=1