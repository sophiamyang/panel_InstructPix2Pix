import io

import hvplot.pandas
import numpy as np
import panel as pn
import param
import PIL
import requests
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline

pn.extension(template='bootstrap')
pn.state.template.main_max_width = '690px'
pn.state.template.accent_base_color = '#F08080'
pn.state.template.header_background = '#F08080'


# Model
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")


def new_image(prompt, image, img_guidance, guidance, steps):
    edit = pipe(
        prompt,
        image=image,
        image_guidance_scale=img_guidance,
        guidance_scale=guidance,
        num_inference_steps=steps,
    ).images[0]
    return edit


# Panel widgets
file_input = pn.widgets.FileInput(width=600)
prompt = pn.widgets.TextInput(
    value="", placeholder="Enter image editing instruction here...", width=600
)
img_guidance = pn.widgets.DiscreteSlider(
    name="Image guidance scale", options=list(np.arange(1, 10.5, 0.5)), value=1.5
)
guidance = pn.widgets.DiscreteSlider(
    name="Guidance scale", options=list(np.arange(1, 10.5, 0.5)), value=7
)
steps = pn.widgets.IntSlider(
    name="Inference Steps", start=1, end=100, step=1, value=20
)
run_button = pn.widgets.Button(name="Run!", width=600)


# define global variables to keep track of things
convos = []  # store all panel objects in a list
image = None
filename = None

def normalize_image(value, width):
    """
    normalize image to RBG channels and to the same size
    """
    b = io.BytesIO(value)
    image = PIL.Image.open(b).convert("RGB")
    aspect = image.size[1] / image.size[0]
    height = int(aspect * width)
    return image.resize((width, height), PIL.Image.ANTIALIAS)

def get_conversations(_, img, img_guidance, guidance, steps, width=600):
    """
    Get all the conversations in a Panel object
    """
    global image, filename
    prompt_text = prompt.value
    prompt.value = ""
    
    # if the filename changes, open the image again
    if filename != file_input.filename:
        filename = file_input.filename
        image = normalize_image(file_input.value, width)
        convos.clear()
        
    if prompt_text:
        # generate new image
        image = new_image(prompt_text, image, img_guidance, guidance, steps)
        convos.append(pn.Row("\U0001F60A", pn.pane.Markdown(prompt_text, width=600)))
        convos.append(pn.Row("\U0001F916", image))
    return pn.Column(*convos)


# bind widgets to functions
interactive_conversation = pn.bind(
    get_conversations, run_button, file_input, img_guidance, guidance, steps
)
interactive_upload = pn.bind(pn.panel, file_input, width=600)

# layout
pn.Column(
    pn.pane.Markdown("## \U0001F60A Upload an image file and start editing!"), 
    pn.Column(file_input, pn.panel(interactive_upload)),
    pn.panel(interactive_conversation, loading_indicator=True),
    prompt,
    pn.Row(run_button),
    pn.Card(img_guidance, guidance, steps, width=600, header="Advance settings"),
).servable(title='Stablel Diffusion InstructPix2pix Image Editing Chatbot')
