import torch, os, time, datetime, colab, postprocessor, progress, importlib
from IPython.display import display
importlib.reload(progress)
importlib.reload(postprocessor)

ESRGAN_install()

def process(ShouldSave, ShouldPreview = True):
    colab.prepare("text2img")
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave and colab.save_settings: postprocessor.save_settings(timestamp, mode="text2img")
    num_iterations = colab.settings['Iterations']
    display("Iterations: 0/%d" % num_iterations, display_id="iterations")
    for i in range(num_iterations):
        colab.image_id = i # needed for progress.py
        generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'] + i)
        progress.reset()
        progress.show()
        image = colab.text2img(
            width=colab.settings['Width'],
            height=colab.settings['Height'],
            prompt=colab.settings['Prompt'],
            negative_prompt=colab.settings['NegativePrompt'],
            guidance_scale=colab.settings['GuidanceScale'],
            num_inference_steps=colab.settings['Steps'],
            generator=generator,
            callback=progress.callback if ShouldPreview else None,
            callback_steps=20).images[0]
        progress.show(image)
        imageName = "%d_%d" % (timestamp, i)
        if ShouldSave:
            image = upscale(image)
            path = postprocessor.save_gdrive(image, imageName)
            print("Saved to " + path)
            postprocessor.post_process(image, imageName)
        display("Iterations: %d/%d" % (i + 1,  num_iterations), display_id="iterations")
        
        
def ESRGAN_install():
  if CLEAR_SETUP_LOG: from IPython.display import clear_output; clear_output()
  if not os.path.exists('/content/Real-ESRGAN'):
    !git clone https://github.com/sberbank-ai/Real-ESRGAN
    !pip install -r Real-ESRGAN/requirements.txt
#     !wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x2.pth -O /content/Real-ESRGAN/weights/RealESRGAN_x2.pth
    !wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x4.pth -O /content/Real-ESRGAN/weights/RealESRGAN_x4.pth
#     !wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x8.pth -O /content/Real-ESRGAN/weights/RealESRGAN_x8.pth
  %cd Real-ESRGAN
  from realesrgan import RealESRGAN
  clear_output()
  device = torch.device('cuda')
  global UPSCALE_AMOUNT
  if not os.path.exists(f'/content/Real-ESRGAN/weights/RealESRGAN_x{UPSCALE_AMOUNT}.pth'):
    def closest_value(input_list, input_value):
      difference = lambda input_list : abs(input_list - input_value)
      res = min(input_list, key=difference)
      return res
    nearest_value = closest_value([2,4,8],UPSCALE_AMOUNT)
    print(f'For Real-ESRGAN upscaling only 2, 4, and 8 are supported. Choosing the nearest Value: {nearest_value}')
    UPSCALE_AMOUNT = nearest_value

  model = RealESRGAN(device, scale = UPSCALE_AMOUNT)
  model.load_weights(f'/content/Real-ESRGAN/weights/RealESRGAN_x{UPSCALE_AMOUNT}.pth')
  %cd /content/
  if CLEAR_SETUP_LOG: from IPython.display import clear_output; clear_output()
    
# Diffuse Function
def upscale(image):
    try:
      from realesrgan import RealESRGAN
    except ModuleNotFoundError:
      if not os.path.exists('/content/Real-ESRGAN'):
        ESRGAN_install()
        %cd /content/Real-ESRGAN
        from realesrgan import RealESRGAN
        %cd /content
    device = torch.device('cuda')
    model = RealESRGAN(device, scale = UPSCALE_AMOUNT)
    try:
      model.load_weights(f'/content/Real-ESRGAN/weights/RealESRGAN_x{UPSCALE_AMOUNT}.pth')
    except FileNotFoundError:
#       !wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x2.pth -O /content/Real-ESRGAN/weights/RealESRGAN_x2.pth
      !wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x4.pth -O /content/Real-ESRGAN/weights/RealESRGAN_x4.pth
#       !wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x8.pth -O /content/Real-ESRGAN/weights/RealESRGAN_x8.pth
      model.load_weights(f'/content/Real-ESRGAN/weights/RealESRGAN_x4.pth')
    sr_image = model.predict(np.array(image))
    return sr_image

