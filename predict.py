import os
import sys
import cv2
import time
import torch
import numpy as np
import skvideo.io
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from io import BytesIO, StringIO
from PIL import Image
import sys, json, base64

from model import LinkNet34


v_file = sys.argv[-1]
model_path = './data/models/resnet34_002_cpt.pth'

batch = 8
THRES_VEH = 0.5
THRES_ROAD = 0.5

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8") 

class LyftTestDataset(Dataset):
    def __init__(self, v_file, img_transform=None):
        self.video = skvideo.io.vread(v_file)
        self.img_transform = img_transform

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, idx):
        img = self.video[idx, :, :, :]
        img = cv2.copyMakeBorder(img,4,4,0,0,cv2.BORDER_REFLECT)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

test_dataset = LyftTestDataset(v_file, transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LinkNet34(3, 3).to(device)
state = torch.load(model_path)
model.load_state_dict(state)

answer_key = {}
frame = 1
for data in test_loader:
    pred = model(data.type(torch.cuda.FloatTensor))
    pred = pred.cpu().data.numpy()
    for i in range(pred.shape[0]):
        img = pred[i, :, 4:604, :]
        print(img.shape)
        binary_car_result = np.where(img[2,:,:]>THRES_VEH,1,0).astype('uint8')
        binary_road_result = np.where(img[1,:,:]>THRES_ROAD,1,0).astype('uint8')
        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        frame+=1

print (json.dumps(answer_key))

#with open('preds.json', 'w') as outfile:
    #json.dump(answer_key, outfile)
