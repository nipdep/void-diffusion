import patcher, torch, random, time
from IPython import display
from IPython.display import HTML
model_name = ""
ready = False
tokenizer = None
text2img = None
img2img = None
inpaint = None
settings = { }
save_directory = "AI-Gen"
save_settings = True
image_id = 0
current_mode = ""
def get_current_image_seed():
    global settings, image_id
    return settings['InitialSeed'] + image_id
def get_current_image_uid():
    return "text2img-%d" % get_current_image_seed()
def init(ModelName):
    global model_name, ready, text2img, img2img, inpaint
    model_name = ModelName
    settings['ModelName'] = ModelName
    patcher.patch()
    if not torch.cuda.is_available():
        print("No GPU found. If you are on Colab, go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        print("Initializing model -> " + model_name + ":")
        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
            from transformers import AutoTokenizer
            rev_dict = {
                'fp16' : ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1", "CompVis/stable-diffusion-v1-4"],
                'diffusers-115k' : ["naclbit/trinart_stable_diffusion_v2",],
                'main' : ["hakurei/waifu-diffusion", "nitrosocke/Arcane-Diffusion", "nitrosocke/archer-diffusion", "nitrosocke/elden-ring-diffusion", "nitrosocke/spider-verse-diffusion", "nitrosocke/modern-disney-diffusion", "hakurei/waifu-diffusion", "lambdalabs/sd-pokemon-diffusers", "yuk/fuyuko-waifu-diffusion", "AstraliteHeart/pony-diffusion", "nousr/robo-diffusion", "DGSpitzer/Cyberpunk-Anime-Diffusion", "sd-dreambooth-library/herge-style"]
            }
            
            for r, ml in rev_dict.items():
                if model_name in ml:
                    rev = r
                    break
            else:
                rev = "main"
            print("revision > ", rev)
#             rev = "main" # "diffusers-115k" if model_name == "naclbit/trinart_stable_diffusion_v2" else "fp16"
            text2img = StableDiffusionPipeline.from_pretrained(model_name, revision=rev, torch_dtype=torch.float16).to("cuda:0")
            img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
            inpaint = StableDiffusionInpaintPipeline(**text2img.components)
            print("Done.")
            ready = True
            from IPython.display import clear_output; clear_output()
            display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected." % model_name))
        except Exception as e:
            print("Failed to initialize model %s with error %s" % (model_name, e))

def prepare(mode):
    global current_mode, settings
    if 'Seed' not in settings:
        print("Please set your settings first.")
        return
    if settings['Seed'] == 0:
        random.seed(int(time.time_ns()))
        settings['InitialSeed'] = random.getrandbits(64)
    else:
        settings['InitialSeed'] = settings['Seed']
    current_mode = mode
    torch.cuda.empty_cache()
