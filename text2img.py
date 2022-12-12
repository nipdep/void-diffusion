import torch, os, time, datetime, colab, postprocessor, progress, importlib
from IPython.display import display
importlib.reload(postprocessor)
def process(ShouldSave, ShouldPreview = True):
    colab.prepare("text2img")
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave and colab.save_settings: postprocessor.save_settings(timestamp, mode="text2img")
    num_iterations = colab.settings['Iterations']
    display("Iterations: 0/%d" % num_iterations, display_id="iterations")
    for i in range(num_iterations):
        colab.image_id = i # needed for progress.py
        print("Seed: %d" % colab.get_current_image_seed())
        generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'] + i)
        image = colab.text2img(
            width=colab.settings['Width'],
            height=colab.settings['Height'],
            prompt=colab.settings['Prompt'],
            negative_prompt=colab.settings['NegativePrompt'],
            guidance_scale=colab.settings['GuidanceScale'],
            num_inference_steps=colab.settings['Steps'],
            generator=generator,
            callback=progress.callback,
            callback_steps=10).images[0]
        if ShouldPreview:
            display(image, display_id=str(i))
        if ShouldSave:
            imageName = "%d_%d" % (timestamp, i)
            path = postprocessor.save_gdrive(image, imageName)
            print("Saved to " + path)
            postprocessor.post_process(image, imageName)
        display("Iterations: %d/%d" % (i / num_iterations), display_id="iterations")