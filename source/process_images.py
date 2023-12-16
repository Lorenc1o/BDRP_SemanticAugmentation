import model as pl
import os
import time
import numpy as np
from PIL import Image

dtype =[
    ("time", float),
    ("prompt_type", str),
    ("adapter", str)
]

measures = np.zeros(0, dtype=dtype)

if __name__ == '__main__':
    pipe = pl.Model("depth-midas")
    pipe_canny = pl.Model("canny")

    dataset_loc = "data/images/"

    #for g in guidance_scale:
        #for a in adapter_conditioning_scale:
        #    for f in adapter_conditioning_factor:
        #        for n in num_inference_steps:
    prompt = "Realistic grey robotic arm with small grippers, anchored to a white table and a small cubic object on the table. White plain background and soft lighting. HQ and highly detailed."
    negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured, strong reflections, strong lighting"
    style = "Real"

    vague_prompt = "Realistic robotic arm, anchored to a table and an object on the table. HQ and highly detailed."
    
    for filename in os.listdir(dataset_loc):
        if filename.endswith(".png"):
            print(filename)
            image = Image.open(dataset_loc + filename)

            start = time.time()
            gen_images = pipe.run(image, prompt, negative_prompt, adapter_name="depth-midas", num_inference_steps=25)[1]
            end = time.time()
            measures = np.append(measures, np.array([(end-start, "detailed", "depth-midas")], dtype=dtype))
            gen_images.save("data/midas_generated/" + filename)

            start = time.time()
            gen_images = pipe_canny.run(image, prompt, negative_prompt, adapter_name="canny", num_inference_steps=25)[1]
            end = time.time()
            measures = np.append(measures, np.array([(end-start, "detailed", "canny")], dtype=dtype))
            gen_images.save("data/canny_generated/" + filename)

            start = time.time()
            gen_images = pipe.run(image, vague_prompt, negative_prompt, adapter_name="depth-midas", num_inference_steps=25)[1]
            end = time.time()
            measures = np.append(measures, np.array([(end-start, "vague", "depth-midas")], dtype=dtype))
            gen_images.save("data/midas_generated_vague/" + filename)

            start = time.time()
            gen_images = pipe_canny.run(image, vague_prompt, negative_prompt, adapter_name="canny", num_inference_steps=25)[1]
            end = time.time()
            measures = np.append(measures, np.array([(end-start, "vague", "canny")], dtype=dtype))
            gen_images.save("data/canny_generated_vague/" + filename)
        else:
            continue

    # Save measures as csv
    np.savetxt("data/measures.csv", measures, delimiter=",", fmt="%s")