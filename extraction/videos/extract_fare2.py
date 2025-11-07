from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import open_clip
import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()
base_folder = os.getenv('FRAME_DIR')

np.set_printoptions(threshold=np.inf)

model_name = "LEAF-CLIP/OpenCLIP-ViT-H-rho50-k1-constrained-FARE2"
processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(processor_name)

model = model.to('cuda')

out_folder = 'features/Vit-H-14_rho50_k1_constrained_FARE2'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

def feature(path):
    image = preprocess(Image.open(path)).unsqueeze(0).to('cuda')
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return np.array2string(image_features.cpu().numpy().flatten(), max_line_width=1000000, separator=',')[1:-1]


for vid in range(1, 7476):
    video = str(vid).zfill(5)
    outfile = out_folder + '/' + video + '.tsv'

    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        print('skipping', video)
        continue

    image_folder = base_folder + video

    if not os.path.exists(image_folder):
      print(video, 'not found')
      continue


    all_frames = 

    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f)) and f.endswith('.png')]

    with open(outfile, 'w') as f:
        for img in image_files:
            try:
                f.write(img + '\t' + feature(join(image_folder, img)) + '\n')
            except:
                print('error while processing', img)
    print('done', video)
