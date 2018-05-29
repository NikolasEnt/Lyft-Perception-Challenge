import os
import sys
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from http.server import BaseHTTPRequestHandler, HTTPServer

from io import BytesIO, StringIO
from PIL import Image
import sys, json, pybase64

from model import LinkNet34

from joblib import Parallel, delayed

model_path = './data/models/resnet34_test_cpt.pth'

batch = 4
THRES_VEH = 0.01
THRES_ROAD = 0.5

class Dilation(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.pad = kernel_size//2
        if kernel_size == 3:
            kernel = torch.from_numpy(np.array([[0,1,0],
                                                [1,1,1],
                                                [0,1,0]]))
        elif kernel_size == 5:
            kernel = torch.from_numpy(np.array([[0,0,1,0,0],
                                                [0,1,1,1,0],
                                                [1,1,1,1,1],
                                                [0,1,1,1,0],
                                                [0,0,1,0,0]]))
        else:
            kernel = torch.ones(kernel_size, kernel_size)
            n = kernel_size**2
        self.kernel = (kernel.unsqueeze(0).unsqueeze(0)\
                       .type(torch.cuda.FloatTensor))
        self.zero_b = torch.tensor(0).type(torch.cuda.ByteTensor)
        self.one_b = torch.tensor(1).type(torch.cuda.ByteTensor)
    
    def forward(self, x):
        x = torch.nn.functional.conv2d(torch.unsqueeze(x, 1),
                                         self.kernel, padding=self.pad)
        x = torch.squeeze(x)
        x = torch.where(x>0, self.one_b, self.zero_b)
        return x

class Erosion(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.pad = kernel_size//2
        if kernel_size == 3:
            kernel = torch.from_numpy(np.array([[0,1,0],
                                                [1,1,1],
                                                [0,1,0]]))
            n = 5
        elif kernel_size == 5:
            kernel = torch.from_numpy(np.array([[0,0,1,0,0],
                                                [0,1,1,1,0],
                                                [1,1,1,1,1],
                                                [0,1,1,1,0],
                                                [0,0,1,0,0]]))
            n = 13
        else:
            kernel = torch.ones(kernel_size, kernel_size)
            n = kernel_size**2
        self.kernel = (kernel.unsqueeze(0).unsqueeze(0)\
                       .type(torch.cuda.FloatTensor)/n)
        self.zero_b = torch.tensor(0).type(torch.cuda.ByteTensor)
        self.one_b = torch.tensor(1).type(torch.cuda.ByteTensor)
        self.one_eps = 1 - 1e-4
    
    def forward(self, x):
        x = torch.nn.functional.conv2d(torch.unsqueeze(x, 1),
                                         self.kernel, padding=self.pad)
        x = torch.squeeze(x)
        x = torch.where(x>=self.one_eps, self.one_b, self.zero_b)
        return x

class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.zero = torch.tensor(0).type(torch.cuda.FloatTensor)
        self.one = torch.tensor(1).type(torch.cuda.FloatTensor)
        
        self.zero_b = torch.tensor(0).type(torch.cuda.ByteTensor)
        self.one_b = torch.tensor(1).type(torch.cuda.ByteTensor)
        self.erosion = Erosion(5)
        self.dilation = Dilation(3)

    def forward(self, x):
        x = x[:, :, 4:604, :]
        car = torch.where(x[:,2,:,:]>THRES_VEH, self.one, self.zero)
        road = torch.where(x[:,1,:,:]>THRES_ROAD, self.one_b, self.zero_b)
        #other = torch.where(x[:,0,:,:]>THRES_ROAD, self.zero, self.one)
        #car = car+other-road
        #road = other-car
        #car = torch.where(car>0, self.one_b, self.zero_b)
        #road = torch.where(road>0, self.one_b, self.zero_b)
        car = self.dilation(car)#.type(torch.cuda.FloatTensor)
        #road = self.erosion(road).type(torch.cuda.FloatTensor)
        #road = self.dilation(road).type(torch.cuda.FloatTensor)
        #road = self.erosion(road)
        #car = self.erosion(car)
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

device = torch.device("cuda:0") 
model = LinkNet34(3, 3).to(device)
state = torch.load(model_path)
model.load_state_dict(state)
model.eval()
postprocess = PostProcess()

def predict(v_file):
    test_dataset = LyftTestDataset(v_file, ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                             num_workers=2, pin_memory=True)
    answer_key = {}
    frame = 1
    with torch.no_grad():
        with Parallel(n_jobs=batch, backend="threading") as parallel:
            for data in test_loader:
                pred = model(data.to(device))
                pred = postprocess(pred)
                res = parallel(delayed(process_pred)(pred[0][i, :, :], pred[1][i, :, :]) for i in range(pred[0].shape[0]))
                answer_key.update({(i+frame):enc for (i, enc) in enumerate(res)})
                frame+=len(res)
    return answer_key


#Create custom HTTPRequestHandler class
class FunHTTPRequestHandler(BaseHTTPRequestHandler):
  #handle GET command
  def do_GET(self):
    try:
        #send code 200 response
        self.send_response(200)
        #send header first
        self.send_header('Content-type','text-html')
        self.end_headers()
        answer_key = predict(self.path)
        self.wfile.write(bytes(json.dumps(answer_key), "utf-8"))
        return

    except IOError:
      self.send_error(404, 'file not found')
  
def run():
  print('http server is starting...')
  #ip and port of servr
  #by default http server port is 8081
  server_address = ('127.0.0.1', 8081)
  httpd = HTTPServer(server_address, FunHTTPRequestHandler)
  print('http server is running...')
  httpd.serve_forever()

run()
