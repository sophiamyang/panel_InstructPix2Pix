import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import panel as pn

pn.extension()
import hvplot.pandas
import io
import param
import numpy as np

# Model
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")


def new_image(prompt, image, image_slider, guidance_slider, step):
    edit = pipe(
        prompt,
        image=image,
        image_guidance_scale=image_slider,
        guidance_scale=guidance_slider,
        num_inference_steps=step,
    ).images[0]
    return edit


# Panel widgets
fi = pn.widgets.FileInput()
inp = pn.widgets.TextInput(
    value="", placeholder="Enter image editing instruction here...", width=600
)
button_conversation = pn.widgets.Button(name="Run!", width=600)
image_slider = pn.widgets.DiscreteSlider(
    name="Image guidance scale", options=list(np.arange(1, 10.5, 0.5)), value=1.5
)
guidance_slider = pn.widgets.DiscreteSlider(
    name="Guidance scale", options=list(np.arange(1, 10.5, 0.5)), value=7
)
step = pn.widgets.IntSlider(
    name="Image guidance scale", start=1, end=100, step=1, value=20
)

# define global variables to keep track of things
convos = []  # store all panel objects in a list
image = None
filename = None


def get_conversations(_, img, image_slider, guidance_slider, step, width=600):
    """
    Get all the conversations in a Panel object
    """
    global image, filename
    prompt = inp.value
    inp.value = ""
    # if the filename changes, open the image again
    if filename != fi.filename:
        filename = fi.filename
        b = io.BytesIO()
        fi.save(b)
        image = PIL.Image.open(b).convert("RGB")
        # resize image if it's too large
        aspect = image.size[1] / image.size[0]
        height = int(aspect * width)
        image = image.resize((width, height), PIL.Image.ANTIALIAS)
        convos.clear()
    if prompt:
        # generate new image
        image = new_image(prompt, image, image_slider, guidance_slider, step)
        convos.append(pn.Row("\U0001F60A", pn.pane.Markdown(prompt, width=600)))
        convos.append(pn.Row("\U0001F916", image))
    return pn.Column(*convos)


# bind widgets to functions
interactive_conversation = pn.bind(
    get_conversations, button_conversation, fi, image_slider, guidance_slider, step
)
interactive_upload = pn.bind(pn.panel, fi, width=600)

# layout
pn.Column(
    pn.Column(fi, pn.panel(interactive_upload)),
    pn.panel(interactive_conversation, loading_indicator=True),
    inp,
    pn.Row(button_conversation),
    pn.Card(image_slider, guidance_slider, step, width=600, header="Advance settings"),
).servable()
