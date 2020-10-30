#from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
print(torch.__version__)
#import numpy as np
from lib import _C
#import pdb

def nms_gpu(dets, thresh):
	keep = dets.new(dets.size(0), 1).zero_().int()
	num_out = dets.new(1).zero_().int()
	_C.nms_cuda(keep, dets, num_out, thresh)
	keep = keep[:num_out[0]]
	return keep
