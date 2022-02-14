# PPHM-GAN for multi-stain transformation of pathological images

Official implementation of PPHM-GAN (PAS, PAM, H&E, and Masson generative adversarial networks). 

This repository contains a PyTorch model for reproducing the experiment.

# How To Use PPHM-GAN Models

Prerequisites:
- PyTorch, version 1.6 or higher
- numpy

Please refer to the original paper (under submission) for the training details. 

```
import torch
from models import G, D
G = G() # default options
D = D() # default options

dummy_input_G = torch.zeros(1, 3, 512, 512)
dummy_stain_transformed_to = torch.Tensor([0, 1, 0, 0]).unsqueeze(0) # one-hot encoding
dummy_input_D = torch.zeros(1, 3, 512, 512)

fake_image = G(dummy_input_G, dummy_stain_transformed_to)
out_D = D(dummy_input_D)
```



