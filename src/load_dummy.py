'''
Load model and run dummy image through it for memory management and evaluation

Author: Prathamesh Mandke

Created: 11/01/2019

Output: python3 load_dummy.py  27.90s user 10.68s system 163% cpu 23.549 total

'''

from net import reconnet
import random
import numpy as np

from datetime import timedelta
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main():
    t1 = time.monotonic()
    model = reconnet(in_channels=1, out_channels=46)
    t2 = time.monotonic()

    outp = model(torch.randn(1, 1, 128, 128))

    print("The model summary is as follows \n {}".format(summary(model, (1, 128, 128))))
    print()
    print("Input dimension is {}. Output dimension is {}".format([1, 128, 128], outp.size()))
    print()
    print("Time taken for forward pass of 1 image is {}s".format(timedelta(seconds=t2 - t1)))

main()
