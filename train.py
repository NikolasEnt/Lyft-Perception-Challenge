import os
import cv2
import time
import torch
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from loss import LyftLoss
from model import LinkNet


epochs = 1
batch = 16 
lr = 1e-4
model_path = './data/models/resnet34_016.pth'
load_model_path = None#'./data/models/resnet34_012_04.pth'
encoder='resnet34'
final='softmax'
gamma = 0.25
brightness = 2.0
colors = 0.15
train_dirs = ['data/train/', 'data/dataset/', 'data/carla-capture-20180528/', 'data/data/Train/', 'data/data/Valid/']
val_dirs=['data/data/Test/', 'data/carla-capture-20181305/']


np.random.seed(123)

class AugmentColor(object):
    def __init__(self, gamma, brightness, colors):
        self.gamma = gamma
        self.brightness = brightness
        self.colors = colors

    def __call__(self, img):
        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # randomly shift gamma
            random_gamma = torch.from_numpy(np.random.uniform(1-self.gamma, 1+self.gamma, 1)).type(torch.cuda.FloatTensor)
            img  = img  ** random_gamma

        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # randomly shift brightness
            random_brightness =  torch.from_numpy(np.random.uniform(1/self.brightness, self.brightness, 1))\
                .type(torch.cuda.FloatTensor)
            img  =  img * random_brightness

        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # randomly shift color
            random_colors =  torch.from_numpy(np.random.uniform(1-self.colors, 1+self.colors, 3))\
                .type(torch.cuda.FloatTensor)
            white = torch.ones([np.shape(img)[1], np.shape(img)[2]]).type(torch.cuda.FloatTensor)
            color_image = torch.stack([white * random_colors[i] for i in range(3)], dim=0)
            img  *= color_image

        # saturate
        img  = torch.clamp(img,  0, 1)
        return img

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample).type(torch.cuda.FloatTensor)

train_transform = transforms.Compose([
    ToTensor(),
    AugmentColor(gamma, brightness, colors)
])

val_transform = transforms.Compose([
    ToTensor(),
])

class LyftDataset(Dataset):
    def __init__(self, data_dir, img_transform=None, trg_transform=None, read=True):
        img_dir = os.path.join(data_dir, "CameraRGB")
        trg_dir = os.path.join(data_dir, "CameraSeg")
        img_paths = sorted(os.listdir(img_dir))
        trg_paths = sorted(os.listdir(trg_dir))
        self.img_paths = [os.path.join(img_dir, path) for path in img_paths]
        self.trg_paths = [os.path.join(trg_dir, path) for path in trg_paths]
        if read: 
            self.imgs = [cv2.imread(path) for path in self.img_paths]
            self.trgs = [self._fix_trg(cv2.imread(path)) for path in self.trg_paths]
        self.img_transform = img_transform
        self.trg_transform = trg_transform
        self.read = read
    
    def _fix_trg(self, trg):
        h, w, _ = trg.shape
        mask = np.zeros((h+2, w+2, 1), dtype=np.uint8)
        cv2.floodFill(trg, mask, (w//2, h-1), (0,0,0))
        vehicles = (trg[:, :, 2]==10).astype(np.float)
        road = (trg[:, :, 2]==6).astype(np.float)
        road += (trg[:, :, 2]==7).astype(np.float)
        bg = np.ones(vehicles.shape) - vehicles - road
        return np.stack([bg, road, vehicles], axis=2)

    def __len__(self):
        if self.read:
            return len(self.imgs)
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        if self.read:
            img = self.imgs[idx]
            trg = self.trgs[idx]
        else:
            img = cv2.imread(self.img_paths[idx])
            trg = self._fix_trg(cv2.imread(self.trg_paths[idx]))
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.trg_transform is not None:
            trg = self.trg_transform(trg)
        return img, trg

train_datasets = [LyftDataset(train_dir, train_transform, transforms.ToTensor(), False) for train_dir in train_dirs]
train_dataset = ConcatDataset(train_datasets)
print("Train imgs:", train_dataset.__len__())
val_datasets = [LyftDataset(val_dir, val_transform, transforms.ToTensor(), False) for val_dir in val_dirs]
val_dataset = ConcatDataset(val_datasets)
print("Train imgs:", val_dataset.__len__())

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loss = LyftLoss(bce_w=0, car_w=1.5, other_w=0.25).to(device)
val_loss = LyftLoss(bce_w=0, car_w=1, other_w=0).to(device)
model = LinkNet(3, 3, encoder, final).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

if load_model_path is not None:
    state = torch.load(load_model_path)
    model.load_state_dict(state)
    
def val():
    c_loss = 0
    with torch.no_grad():
        for img, trg in val_loader:
            img = img.type(torch.cuda.FloatTensor)
            trg = trg.type(torch.cuda.FloatTensor)
            pred = model(img)
            loss = val_loss(pred, trg)
            c_loss += loss.item()
        c_loss /= val_dataset.__len__()
    return c_loss

def train(epochs):
    losses = []
    best_loss = val()
    print("Start val loss:", best_loss)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        s_time = time.time()
        for img, trg in train_loader:
            # get the inputs
            img = img.type(torch.cuda.FloatTensor)
            trg = trg.type(torch.cuda.FloatTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(img)
            loss = train_loss(pred, trg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= train_dataset.__len__()
        val_s = val()
        val_s_f = round((2-val_s*batch)/2, 5)
        print("Epoch:", epoch+1, "train loss:", round(running_loss, 5), "val loss", val_s,
              "val score:", val_s_f,
              "time:", round(time.time()-s_time, 2), "s")
        if val_s < best_loss:
            torch.save(model.state_dict(), model_path[:-4]+'_cpt_'+str(val_s_f)+model_path[-4:])
            best_loss = val_s
            print("Checkpoint saved")
        losses.append([running_loss, val])
        running_loss = 0.0
    torch.save(model.state_dict(), model_path[:-4]+'_res_'+str(val_s_f)+model_path[-4:])
    print(losses)

torch.cuda.synchronize()

train(50)
lr=1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
train(10)
lr=1e-6
optimizer = optim.Adam(model.parameters(), lr=lr)
train(10)


print('Finished Training')
torch.save(model.state_dict(), model_path)

