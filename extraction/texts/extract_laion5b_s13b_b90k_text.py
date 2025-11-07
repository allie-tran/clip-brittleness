import torch
from PIL import Image
import open_clip
import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)

model_name = 'laion5b_s13b_b90k'
model_type = 'xlm-roberta-base-ViT-B-32'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=model_name, device=device)
tokenizer = open_clip.get_tokenizer(model_type)

with open("queries/permutations.csv") as file:
    lines = [line.rstrip().split(",")[2] for line in file]

lines = lines[1:]

with open("features/" + model_name + "_text.csv", 'w') as f:
    for line in tqdm(lines):
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = model.encode_text(tokenizer(line).to(device))
            features /= features.norm(dim=-1, keepdim=True)
            s = np.array2string(features.cpu().numpy().flatten(), max_line_width=100000, separator=',')[1:-1]
            f.write(s)
            f.write("\n")


