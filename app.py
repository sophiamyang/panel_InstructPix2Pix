import io

import numpy as np
import panel as pn
import param
import PIL
import requests
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline

pn.extension("texteditor", template="bootstrap", sizing_mode="stretch_width")

pn.state.template.param.update(
    main_max_width="690px",
    header_background="#F08080",
)

model_id = "timbrooks/instruct-pix2pix"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "pipe" in pn.state.cache:
    pipe = pn.state.cache["pipe"]
else:
    pipe = pn.state.cache[
        "pipe"
    ] = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(
        device
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet.to(memory_format=torch.channels_last)


def normalize_image(value, width):
    """
    normalize image to RBG channels and to the same size
    """
    b = io.BytesIO(value)
    image = PIL.Image.open(b).convert("RGB")
    aspect = image.size[1] / image.size[0]
    height = int(aspect * width)
    return image.resize((width, height), PIL.Image.LANCZOS)


def new_image(prompt, image, img_guidance, guidance, steps, width=600):
    edit = pipe(
        prompt,
        image=image,
        image_guidance_scale=img_guidance,
        guidance_scale=guidance,
        num_inference_steps=steps,
    ).images[0]
    return edit


file_input = pn.widgets.FileInput(width=600)

prompt = pn.widgets.TextEditor(
    value="",
    placeholder="Enter image editing instruction here...",
    height=160,
    toolbar=False,
)
img_guidance = pn.widgets.DiscreteSlider(
    name="Image guidance scale", options=list(np.arange(1, 10.5, 0.5)), value=1.5
)
guidance = pn.widgets.DiscreteSlider(
    name="Guidance scale", options=list(np.arange(1, 10.5, 0.5)), value=7
)
steps = pn.widgets.IntSlider(name="Inference Steps", start=1, end=100, step=1, value=20)
run_button = pn.widgets.Button(name="Run!")

widgets = pn.Row(
    pn.Column(prompt, run_button, margin=5),
    pn.Card(
        pn.Column(img_guidance, guidance, steps), title="Advanced settings", margin=10
    ),
    width=600,
)

# define global variables to keep track of things
convos = []  # store all panel objects in a list
image = None
filename = None


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

    # if there is a prompt run output
    if prompt_text:
        image = new_image(prompt_text, image, img_guidance, guidance, steps)
        convos.extend(
            [
                pn.Row(pn.panel("\U0001F60A", width=10), prompt_text, width=600),
                pn.Row(
                    pn.panel(image, align="end", width=500),
                    pn.panel("\U0001F916", width=10),
                    align="end",
                ),
            ]
        )
    return pn.Column(*convos, margin=15, width=575)


# bind widgets to functions
interactive_upload = pn.panel(
    pn.bind(pn.panel, file_input, width=575, min_height=400, margin=15)
)

interactive_conversation = pn.panel(
    pn.bind(get_conversations, run_button, file_input, img_guidance, guidance, steps),
    loading_indicator=True,
)


# layout
pn.Column(
    "## \U0001F60A Upload an image file and start editing!",
    file_input,
    interactive_upload,
    interactive_conversation,
    widgets,
).servable(title="Panel Stable Diffusion InstructPix2pix Image Editing Chatbot")
