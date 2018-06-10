import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AugmentColor(object):
    def __init__(self, gamma, brightness, colors):
        self.gamma = gamma
        self.brightness = brightness
        self.colors = colors

    def __call__(self, img):
        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # Randomly shift gamma
            random_gamma = torch.from_numpy(np.random.uniform(\
                                1 - self.gamma, 1+self.gamma, 1))\
                                .type(torch.cuda.FloatTensor)
            img  = img  ** random_gamma

        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # Randomly shift brightness
            random_brightness =  torch.from_numpy(np.random.uniform(1\
                    / self.brightness, self.brightness, 1))\
                    .type(torch.cuda.FloatTensor)
            img  =  img * random_brightness

        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            # Randomly shift color
            random_colors =  torch.from_numpy(np.random.uniform(1 -\
                    self.colors, 1+self.colors, 3))\
                    .type(torch.cuda.FloatTensor)
            white = torch.ones([np.shape(img)[1], np.shape(img)[2]])\
                                .type(torch.cuda.FloatTensor)
            color_image = torch.stack([white * random_colors[i]
                                      for i in range(3)], dim=0)
            img  *= color_image

        # Saturate
        img  = torch.clamp(img,  0, 1)
        return img

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample).type(torch.cuda.FloatTensor)


class LyftDataset(Dataset):
    def __init__(self, data_dir, hood_path, top, bottom,
                 img_transform=None, trg_transform=None, read=True):
        img_dir = os.path.join(data_dir, "CameraRGB")
        trg_dir = os.path.join(data_dir, "CameraSeg")
        img_paths = sorted(os.listdir(img_dir))
        trg_paths = sorted(os.listdir(trg_dir))
        self.img_paths = [os.path.join(img_dir, path) for path
                          in img_paths]
        self.trg_paths = [os.path.join(trg_dir, path) for path
                          in trg_paths]
        if read: 
            self.imgs = [cv2.imread(path) for path in self.img_paths]
            self.trgs = [self._fix_trg(cv2.imread(path)) for path
                         in self.trg_paths]
        self.img_transform = img_transform
        self.trg_transform = trg_transform
        self.read = read
        self.hood = np.load(hood_path)  # Load the hood mask
        self.top = top
        self.bottom = bottom

    def _fix_trg(self, trg):
        vehicles = (trg[:, :, 2]==10).astype(np.float)
        vehicles = np.logical_and(vehicles, self.hood)
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
        img = img[self.top:self.bottom, :, :]
        trg = trg[self.top:self.bottom, :, :]
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.trg_transform is not None:
            trg = self.trg_transform(trg)
        return img, trg


def get_kernel(n, square=False):
    assert n in [3, 5], "Incorrect kernel size"
    # TODO Make it nice and usable for arbitrary kernel size
    if square:
            kernel = torch.ones(n, n)
            el = n**2
    else:
        if n == 3:
            kernel = torch.from_numpy(np.array([[0,1,0],
                                                [1,1,1],
                                                [0,1,0]]))
            el = 5
        elif n == 5:
            kernel = torch.from_numpy(np.array([[0,0,1,0,0],
                                                [0,1,1,1,0],
                                                [1,1,1,1,1],
                                                [0,1,1,1,0],
                                                [0,0,1,0,0]]))
            el = 13
    return kernel, el
    

class Dilation(torch.nn.Module):
    def __init__(self, kernel_size, kernel_square):
        super().__init__()
        self.pad = kernel_size//2
        if kernel_size in [3, 5]:
            kernel, _ = get_kernel(kernel_size, kernel_square)
        else:
            # Use a square kernel, if does not match kernel size
            kernel = torch.ones(kernel_size, kernel_size)
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

class Erosion(torch.nn.Module):
    def __init__(self, kernel_size, kernel_square):
        super().__init__()
        self.pad = kernel_size//2
        if kernel_size in [3, 5]:
            kernel, n = get_kernel(kernel_size, kernel_square)
        else:
            kernel = torch.ones(kernel_size, kernel_size)
            n = kernel_size**2
        self.kernel = (kernel.unsqueeze(0).unsqueeze(0)\
                       .type(torch.cuda.FloatTensor) / n)
        self.zero_b = torch.tensor(0).type(torch.cuda.ByteTensor)
        self.one_b = torch.tensor(1).type(torch.cuda.ByteTensor)
        self.one_eps = 1 - 1e-4

    def forward(self, x):
        x = torch.nn.functional.conv2d(torch.unsqueeze(x, 1),
                                       self.kernel, padding=self.pad)
        x = torch.squeeze(x)
        x = torch.where(x>=self.one_eps, self.one_b, self.zero_b)
        return x
