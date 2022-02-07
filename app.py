import torch
from PIL import Image
from transformers import (AutoTokenizer, VisionEncoderDecoderModel,
                          ViTFeatureExtractor)
from data_loaders import modify_dataset
import pandas as pd
from tqdm import tqdm
import gradio as gr

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "distilgpt2"
model_checkpoint = "gagan3012/ViTGPT2_vizwiz"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

def predict(image):
    clean_text = lambda x: x.replace("<|endoftext|>", "").split("\n")[0]
    sample = feature_extractor(image, return_tensors="pt").pixel_values.to(device) 
    caption_ids = model.generate(sample, max_length=50)[0]
    caption_text = clean_text(tokenizer.decode(caption_ids))
    return caption_text
  
inputs = [
    gr.inputs.Image(type="pil", label="Original Image")
]

outputs = [
    gr.outputs.Textbox(label = 'Caption')
]

title = "Image Captioning using ViT + GPT2"
description = "ViT and GPT2 are used to generate Image Caption for the uploaded images"
article = " <a href='https://huggingface.co/gagan3012/ViTGPT2_vizwiz'>Model Repo on Hugging Face Model Hub</a>"
examples = [
    ["people-walking-street-pedestrian-crossing-traffic-light-city.jpeg"],
    ["elonmusk.jpeg"]
]

gr.Interface(
    predict,
    inputs,
    outputs,
    title=title,
    description=description,
    article=article,
    examples=examples,
    theme="huggingface",
).launch(debug=True, enable_queue=True)
