import os
import random
from typing import Any
from tqdm import tqdm

from torch.utils import data
import  torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')

def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    try:
      import accimage
      return accimage_loader(path)
    except ImportError:
      return pil_loader(path)
  else:
    return pil_loader(path)
    
class PPHM(Dataset):
    def __init__(self, dataset_dir, index_filename=None, transform=None, mode='train', load_in_mem=False, loader=pil_loader, **kwargs):
        self.stain2label = {'PAS': 0, 'PAM': 1, 'HE': 2, 'MT': 3}
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.mode = mode
        self.load_in_mem = load_in_mem
        self.loader = loader

        if index_filename is None:
          index_filename = f'PPHM_{mode}_imgs.pickle'

        import pickle
        if os.path.exists(index_filename):
            print('Loading pre-saved Index file %s...' % index_filename)
            with open(index_filename, 'rb') as f:
                data = pickle.load(f)
            self.imgs = data['imgs']
            self.labels = data['labels']
        else:
            imgs = []
            labels = []
            print('Generating  Index file %s...' % index_filename)
            for stain in os.listdir(os.path.join(dataset_dir, mode)):
                for img in os.listdir(os.path.join(dataset_dir, mode, stain)):
                    if os.path.splitext(img)[-1] in IMG_EXTENSIONS:
                        imgs.append(os.path.join(dataset_dir, mode, stain, img))
                        labels.append(self.stain2label[stain])
            data = {'imgs': imgs, 'labels': labels}
            with open(index_filename, 'wb') as f:
                pickle.dump(data, f)
            self.imgs = imgs
            self.labels = labels
        if self.load_in_mem:
           self.img_data = [ self.loader(str(img)) for im in self.imgs ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.load_in_mem:
            image = self.img_data[idx]
        else:
            image = self.loader(str(self.imgs[idx]))
        label = self.labels[idx]
        return self.transform(image), torch.Tensor([label]).to(torch.int64)
    
# Convenience function to centralize all data loaders
def get_data_loaders(dataset, dataset_dir=None, random_flip=False, random_rotate=0.0,  resize=False, batch_size=64, image_size=128, mode='train',
                     num_workers=1, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False, 
                     **kwargs):
    print('Using dataset root location %s' % dataset_dir)
    if dataset=='PPHM':
        which_dataset = PPHM
    else:
      raise NotImplementedError()


    norm_mean, norm_std = [0.5,0.5,0.5], [0.5,0.5,0.5]

    val_transform = []
    if resize:
        val_transform.append(transforms.Resize(image_size))    
    val_transform.append( transforms.RandomCrop(image_size, padding=None))                          
    val_transform = transforms.Compose(val_transform + [
                     transforms.ToTensor(),
                     transforms.Normalize(norm_mean, norm_std)])

    train_transform = []
    if resize:
      train_transform.append(transforms.Resize(image_size))                   
    if random_flip:
      print('Data will be randomly fliped...')
      train_transform.append( transforms.RandomHorizontalFlip())                       
    if random_rotate:    
      train_transform.append( transforms.RandomRotation(degrees=random_rotate))
    train_transform.append( transforms.RandomCrop(image_size, padding=None))                          
    train_transform = transforms.Compose(train_transform + [
                     transforms.ToTensor(),
                     transforms.Normalize(norm_mean, norm_std)])
 
    train_set = which_dataset(dataset_dir=dataset_dir, mode=mode, transform=train_transform, load_in_mem=load_in_mem,
                         **kwargs)
    val_set = which_dataset(dataset_dir=dataset_dir, mode=mode, transform=val_transform, load_in_mem=load_in_mem,
                         **kwargs)

    loaders = []   
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                      'drop_last': drop_last} # Default, drop last incomplete batch
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=shuffle, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                                  shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)
    loaders.append(val_loader)
    return loaders
