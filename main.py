# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
vgg = scipy.io.loadmat(VGG_MODEL)


