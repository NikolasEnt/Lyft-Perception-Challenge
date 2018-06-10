import os
import cv2
import time
import torch
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

from dataprocess import AugmentColor, ToTensor, LyftDataset
from loss import LyftLoss
from model import LinkNet


batch = 16  # 32 if you have 16 GB of VRAM
hood_path = 'hood.npy'  # Path to the saved hood mask
model_path = './data/models/resnet34_001.pth'  # Name for the model save
load_model_path = None  # Load pretrain path
encoder = 'resnet34'  # Encoder type: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
final = 'softmax'  # Output layer type. 'softmax' or 'sigmoid'
# Image augmentation parameters
gamma = 0.35
brightness = 2.0
colors = 0.25

train_dirs = ['data/train/']  # List train dirs here
val_dirs = ['data/val/']  # List dirs with validation datasets

np.random.seed(123)

# Loss components weights
train_bce_w = 0.0
train_car_w = 1.0
train_other_w = 1.0
val_bce_w = 0.0
val_car_w = 1.0
val_other_w = 0.0

# Define ROI:
top = 205
bottom = 525  # bottom-top should be a multiple of 32

train_transform = transforms.Compose([
    ToTensor(),
    AugmentColor(gamma, brightness, colors)
])

val_transform = transforms.Compose([
    ToTensor(),
])


train_datasets = [LyftDataset(dir, hood_path, top, bottom,
                  train_transform, transforms.ToTensor(), False)
                  for dir in train_dirs]
train_dataset = ConcatDataset(train_datasets)

val_datasets = [LyftDataset(dir, hood_path, top, bottom, val_transform,
                transforms.ToTensor(), False) for dir in val_dirs]
val_dataset = ConcatDataset(val_datasets)
print("Train imgs:", train_dataset.__len__())
print("Val imgs:", val_dataset.__len__())

assert torch.cuda.is_available(), "Sorry, no CUDA device found"

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

device = torch.device("cuda")

train_loss = LyftLoss(bce_w=train_bce_w, car_w=train_car_w,
                      other_w=train_other_w).to(device)
val_loss = LyftLoss(bce_w=val_bce_w, car_w=val_car_w,
                    other_w=val_other_w).to(device)

model = LinkNet(3, 3, encoder, final).to(device)
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
        val_s_f = round((2-val_s*batch)/2, 5)  # LB score on val
        print("Epoch:", epoch+1, "train loss:", round(running_loss, 5),
              "val loss", round(val_s, 5), "val score:", val_s_f,
              "time:", round(time.time()-s_time, 2), "s")
        if val_s < best_loss:
            torch.save(model.state_dict(), model_path[:-4] + '_cpt_' \
                       + str(val_s_f) + model_path[-4:])
            best_loss = val_s
            print("Checkpoint saved")
        losses.append([running_loss, val])
        running_loss = 0.0
    # Save the train result
    torch.save(model.state_dict(), model_path[:-4] + '_res_'\
               + str(val_s_f) + model_path[-4:])
    print(losses)

torch.cuda.synchronize()


# Define the train protocol here
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
train(epochs=75)
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
train(epochs=20)
lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr)
train(epochs=10)


# Save the final result
torch.save(model.state_dict(), model_path)
print('Finished Training')
