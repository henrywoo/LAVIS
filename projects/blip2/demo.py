import torch
from PIL import Image

import time

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#display(raw_image.resize((596, 437)))

from lavis.models import load_model_and_preprocess
# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
#model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
model = model.half()

time.sleep(600)

# load sample image
raw_image = Image.open("../../docs/_static/merlion.png").convert("RGB")
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
answer = model.generate({"image": image, "prompt": prompt})
print(answer)
# 'singapore'
answer = model.generate({
    "image": image,
    "prompt": "Question: which city is this? Answer: singapore. Question: why?"})
print(answer)
# 'it has a statue of a merlion'    

# prepare context prompt
context = [
    ("which city is this?", "singapore"),
    ("why?", "it has a statue of a merlion"),
]
question = "where is the name merlion coming from?"
template = "Question: {} Answer: {}."
prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"
print(prompt)
# generate model's response
answer=model.generate({"image": image,"prompt": prompt})
print(answer)
# 'merlion is a portmanteau of mermaid and lion'

b2 = Blip2Inference()
print(b2.predict())
