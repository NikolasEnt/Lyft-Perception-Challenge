import os
import sys
import cv2
import time

import numpy as np
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from io import BytesIO, StringIO
from PIL import Image
import sys, json, pybase64

from model import LinkNet34

from joblib import Parallel, delayed

#def init_yappi():
  #OUT_FILE = './profile'

  #import atexit
  #import yappi

  #print('[YAPPI START]')
  #yappi.set_clock_type('wall')
  #yappi.start()

  #@atexit.register
  #def finish_yappi():
    #print('[YAPPI STOP]')

    #yappi.stop()

    #print('[YAPPI WRITE]')

    #stats = yappi.get_func_stats()

    #for stat_type in ['pstat', 'callgrind', 'ystat']:
      #print('writing {}.{}'.format(OUT_FILE, stat_type))
      #stats.save('{}.{}'.format(OUT_FILE, stat_type), type=stat_type)

    #print('\n[YAPPI FUNC_STATS]')

    #print('writing {}.func_stats'.format(OUT_FILE))
    #with open('{}.func_stats'.format(OUT_FILE), 'w') as fh:
      #stats.print_all(out=fh)

    #print('\n[YAPPI THREAD_STATS]')

    #print('writing {}.thread_stats'.format(OUT_FILE))
    #tstats = yappi.get_thread_stats()
    #with open('{}.thread_stats'.format(OUT_FILE), 'w') as fh:
      #tstats.print_all(out=fh)

    #print('[YAPPI OUT]')

#init_yappi()

c_time = time.time()
v_file = sys.argv[-1]
model_path = './data/models/resnet34_003_cpt2.pth'

batch = 6
THRES_VEH = 0.5
THRES_ROAD = 0.5

class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.zero = torch.tensor(0).type(torch.cuda.ByteTensor)
        self.one = torch.tensor(0).type(torch.cuda.ByteTensor)

    def forward(self, x):
        x = x[:, 1:, 4:604, :]
        car = torch.where(x[:,1,:,:]>THRES_VEH, self.one, self.zero)
        road = torch.where(x[:,0,:,:]>THRES_ROAD, self.one, self.zero)
        return [car, road]

def encode(array):
    buff = cv2.imencode(".png", array)[1]
    return pybase64.b64encode(buff).decode("utf-8") 

def process_pred(car, road):
    return [encode(car.cpu().data.numpy()),
            encode(road.cpu().data.numpy())]

class LyftTestDataset(Dataset):
    def __init__(self, v_file, img_transform=None):
        cap = cv2.VideoCapture(v_file)
        self.video = []
        ret = True
        while ret:
            ret, img = cap.read()
            self.video.append(img)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.video)-1

    def __getitem__(self, idx):
        img = self.video[idx]
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img


test_dataset = LyftTestDataset(v_file, ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                         num_workers=2, pin_memory=True)
device = torch.device("cuda:0") 
model = LinkNet34(3, 3).to(device)
state = torch.load(model_path)
model.load_state_dict(state)
model.eval()

answer_key = {}
frame = 1
postprocess = PostProcess()
with torch.no_grad():
    with Parallel(n_jobs=6, backend="threading") as parallel:
        for data in test_loader:
            pred = model(data.to(device))
            pred = postprocess(pred)
            res = parallel(delayed(process_pred)(pred[0][i, :, :], pred[1][i, :, :]) for i in range(pred[0].shape[0]))
            answer_key.update({(i+frame):enc for (i, enc) in enumerate(res)})
            frame+=pred[0].shape[0]

print(time.time()-c_time)

#print(json.dumps(answer_key))

#with open('preds.json', 'w') as outfile:
    #json.dump(answer_key, outfile)
