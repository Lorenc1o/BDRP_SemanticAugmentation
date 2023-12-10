from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from controlnet_aux.canny import CannyDetector
from controlnet_aux.midas import MidasDetector
import torch
from PIL import Image
import numpy as np


MAX_SEED = np.iinfo(np.int32).max

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Photorealistic",
        "prompt": "photorealistic {prompt} . realistic, high resolution, high quality photo",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    }
]

ADAPTER_REPO_IDS = {
    "canny": "TencentARC/t2i-adapter-canny-sdxl-1.0",
    #"sketch": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
    #"lineart": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
    "midas": "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
    #"depth-zoe": "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
    #"openpose": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
    # "recolor": "TencentARC/t2i-adapter-recolor-sdxl-1.0",
}
ADAPTER_NAMES = list(ADAPTER_REPO_IDS.keys())

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic"

class AugmentationPipeline:
    def __init__(self, device="cuda", model_id="stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors", adapter = "canny"):
        self.device = device
        self.model_id = model_id
        self.weight_name = weight_name
        self.pipe = None
        self.adapter = adapter
        if adapter == "canny":
            self.canny_detector = CannyDetector()
        elif adapter == "midas":
            self.canny_detector = MidasDetector.from_pretrained(
              "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
            )
        self.load_model()

    def load_model(self):
        adapter = T2IAdapter.from_pretrained(ADAPTER_REPO_IDS[self.adapter], torch_dtype=torch.float16, varient="fp16").to(self.device)
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            self.model_id, 
            vae=vae, 
            adapter=adapter, 
            scheduler=euler_a, 
            torch_dtype=torch.float16, 
            variant="fp16", 
        ).to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.load_lora_weights(self.model_id, weight_name=self.weight_name)
        self.pipe.fuse_lora(lora_scale = 0.4)

    def apply_style(self, style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + negative
    
    def augment(self, image, prompt, negative_prompt, style_name, num_inference_steps=50, guidance_scale=7.5, adapter_conditioning_scale=1, adapter_conditioning_factor=1):
        image = self.canny_detector(image)
        prompt, negative_prompt = self.apply_style(style_name, prompt, negative_prompt)
        gen_images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, 
            adapter_conditioning_scale=adapter_conditioning_scale, 
            adapter_conditioning_factor=adapter_conditioning_factor
        ).images[0]
        return gen_images
    
    def augment_from_file(self, filename, prompt, negative_prompt, style_name, num_inference_steps=50, guidance_scale=7.5, adapter_conditioning_scale=1, adapter_conditioning_factor=1):
        image = Image.open(filename)
        return self.augment(image, prompt, negative_prompt, style_name, num_inference_steps, guidance_scale, adapter_conditioning_scale, adapter_conditioning_factor)
    
if __name__ == "__main__":
    aug = AugmentationPipeline()
    img = aug.augment_from_file("data/first_image.png", "Realistic grey robotic arm anchored to a white table and a small object on the table. White plain background and soft lighting", "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured, strong reflections, strong lighting", "Photorrealistic")
    img.save("data/first_image_augmented.png")