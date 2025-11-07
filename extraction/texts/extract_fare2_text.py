import torch
from PIL import Image
import open_clip
import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

np.set_printoptions(threshold=np.inf)

model_name = "LEAF-CLIP/OpenCLIP-ViT-H-rho50-k1-constrained-FARE2"
processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(processor_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
tokenizer = processor.tokenizer

with open("queries/permutations.csv") as file:
    lines = [line.rstrip().split(",")[2] for line in file]

lines = lines[1:]

save_name = 'ViT-H-14_rho50_k1_constrained_FARE2'
with open("features/" + save_name + "_text.csv", 'w') as f:
    for line in tqdm(lines):
        with torch.no_grad(), torch.cuda.amp.autocast():
            inputs = tokenizer([line], padding=True, return_tensors="pt").to(device)
            features = model.get_text_features(**inputs)
            features /= features.norm(dim=-1, keepdim=True)
            s = np.array2string(features.cpu().numpy().flatten(), max_line_width=100000, separator=',', threshold=np.inf)[1:-1]
            f.write(s)
            f.write("\n")


