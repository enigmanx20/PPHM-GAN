[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# PPHM-GAN for multi-stain transformation of pathological images

Official implementation of PPHM-GAN (PAS, PAM, H&E, and Masson generative adversarial networks). An artifical multi-stain to multi-stain transformation platform for digital pathology.

## Usage
First, install PyTorch 1.8 (or later), torchvision, and numpy, as well as additional dependencies by running the snippet below.
```
pip install tqdm, tensorboard
```
Please refer to the original paper (under submission) for the training details. 
Dataset dir is organized as below.
```
datasets_dir:    ~/.../datasets/PPHM
        typical dir tree
        datasets--PPHM--train--PAS
                       |      |-PAM
                       |      |-MT
                       |      |-HE
                       |       
                       -test ---PAS
                              |-PAM
                              |-MT
                              |-HE
```
Edit the config.py and run the training script. Default hyperparameters are optimized for an A100x8 server. Change the 'multi-gpus', 'enable_autocast', 'num_workers', and 'minibatch' parameters for your environment.
```
python train.py
```
Just try the models to see how they work.
```
import torch
from models import Generator, Discriminator
from config import config

G = Generator(**config)
D = Discriminator(**config)

dummy_input_G = torch.zeros(1, 3, 512, 512)
dummy_stain_transformed_to = torch.Tensor([0, 1, 0, 0]).unsqueeze(0) # one-hot encoding
dummy_input_D = torch.zeros(1, 3, 512, 512)

fake_image = G(dummy_input_G, dummy_stain_transformed_to)
out_D = D(dummy_input_D)
```
## Tuning recommendations
Start the learning rate one of 1e-4, 3e-4, or 1e-5. If the training degredes, make smaller the learning rate. After sufficient iterations (e.g. 10M images or 150K iteraions), cosine anealing decay ('cos_lr': True) for a while ('cos_kimg') will improve the quality. If a certain stain transformation pair fails, reduce the stains until CycleGAN-like ont-to-one transformation.
