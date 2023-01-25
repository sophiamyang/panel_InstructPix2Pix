import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import panel as pn
pn.extension()
import hvplot.pandas
import io
import param

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

def new_image(prompt, image):
    edit = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images[0]
    return edit

fi = pn.widgets.FileInput()
inp = pn.widgets.TextInput(value="", placeholder='Enter image editing instruction here...', width=600)
button_conversation = pn.widgets.Button(name="Run!", width=600)
convos = [] # store all panel objects in a list
image = None
filename = None

def get_conversations(_):
    global image, filename
    prompt = inp.value
    inp.value = ''
    # if the filename changes, open the image again
    if filename != fi.filename:
        filename = fi.filename
        b = io.BytesIO()
        fi.save(b)
        image = PIL.Image.open(b)
        convos.clear()
    if prompt:
        # generate new image
        image = new_image(prompt, image)
        convos.append(
            pn.Row('\U0001F60A', pn.pane.Markdown(prompt, width=600))
        )
        convos.append(
            pn.Row('\U0001F916', image)
        )
    return pn.Column(*convos)

interactive_conversation = pn.bind(get_conversations, button_conversation)
interactive_upload = pn.bind(pn.pane.PNG, fi)
pn.Column(
    pn.Column(fi, pn.panel(interactive_upload)),
    pn.panel(interactive_conversation, loading_indicator=True),
    inp,
    pn.Row(button_conversation),
).servable()
   