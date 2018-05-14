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
import sys, json, pybase64

from model import LinkNet34

from joblib import Parallel, delayed


v_file = sys.argv[-1]
model_path = './data/models/resnet34_002_cpt.pth'

batch = 8
THRES_VEH = 0.5
THRES_ROAD = 0.5

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG", compress_level=1)
    return pybase64.b64encode(buff.getvalue()).decode("utf-8") 

def process_pred(img):
    binary_car_result = np.where(img[1,:,:]>THRES_VEH,1,0).astype('uint8')
    binary_road_result = np.where(img[0,:,:]>THRES_ROAD,1,0).astype('uint8')
    return [encode(binary_car_result), encode(binary_road_result)]

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
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False) #num_workers=2, pin_memory=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = LinkNet34(3, 3).to(device)
state = torch.load(model_path)
model.load_state_dict(state)
model.eval()

answer_key = {}
frame = 1
c_time = time.time()
with Parallel(n_jobs=4, backend="threading") as parallel:
    for data in test_loader:
        pred = model(data.type(torch.cuda.FloatTensor))
        pred = pred[:, 1:, 4:604, :].cpu().data.numpy()
        res = parallel(delayed(process_pred)(pred[i, :, :, :]) for i in range(pred.shape[0]))
        answer_key.update({(i+frame):enc for (i, enc) in enumerate(res)})
        frame+=len(res)

print(time.time()-c_time)

#print (json.dumps(answer_key))

with open('preds.json', 'w') as outfile:
    json.dump(answer_key, outfile)
