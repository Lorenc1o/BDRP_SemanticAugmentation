import pipeline as pl
import os
import time
import numpy as np

dtype =[
    ("time", float),
    ("guidance_scale", float),
    ("adapter_conditioning_scale", float),
    ("adapter_conditioning_factor", float),
    ("num_inference_steps", int),
]

measures = np.zeros(0, dtype=dtype)

guidance_scale = [5, 7.5, 10]
adapter_conditioning_scale = [1]
adapter_conditioning_factor = [1]
num_inference_steps = [50]

times = np.zeros(0)

if __name__ == '__main__':
    pipe = pl.AugmentationPipeline(adapter="midas")

    dataset_loc = "data/images/"

    #for g in guidance_scale:
        #for a in adapter_conditioning_scale:
        #    for f in adapter_conditioning_factor:
        #        for n in num_inference_steps:
    prompt = "Realistic grey robotic arm with small grippers, anchored to a white table and a small cubic object on the table. White plain background and soft lighting"
    negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured, strong reflections, strong lighting"
    style = "Photographic"
    for filename in os.listdir(dataset_loc):
        if filename.endswith(".png"):
            print(filename)
            start = time.time()
            gen_images = pipe.augment_from_file(dataset_loc + filename, prompt, negative_prompt, style) #, num_inference_steps=n, guidance_scale=g, adapter_conditioning_scale=a, adapter_conditioning_factor=f)
            end = time.time()
            #measures = np.append(measures, np.array([(end-start, g, a, f, n)], dtype=dtype))
            times = np.append(times, end-start)
            gen_images.save("data/midas_generated/" + filename)
        else:
            continue

    avg_time = np.average(times)
    print("Average time: " + str(avg_time))

    # Save measures as csv
    #np.savetxt("data/measures.csv", measures, delimiter=",")