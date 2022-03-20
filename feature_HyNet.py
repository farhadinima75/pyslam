"""
* This file is part of PYSLAM 
* adapted from https://github.com/DagnyT/hardnet/blob/master/examples/extract_hardnet_desc_from_hpatches_file.py, see licence therein.  
* 
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

# adapted from https://github.com/DagnyT/hardnet/blob/master/examples/extract_hardnet_desc_from_hpatches_file.py 

import config 

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import cv2
import math
import numpy as np

from utils_features import extract_patches_tensor, extract_patches_array, extract_patches_array_cpp


kVerbose = True 


from typing import Dict

import torch
import torch.nn as nn

urls: Dict[str, str] = {}
urls[
    "liberty"
] = "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_LIB.pth"  # pylint: disable
urls[
    "notredame"
] = "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_ND.pth"  # pylint: disable
urls[
    "yosemite"
] = "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_YOS.pth"  # pylint: disable

class FilterResponseNorm2d(nn.Module):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-6,
                 is_bias: bool = True,
                 is_scale: bool = True,
                 is_eps_leanable: bool = False):

        super().__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = nn.parameter.Parameter(torch.ones(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features: int):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super().__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(-torch.ones(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, self.tau)


class HyNet(nn.Module):
    patch_size = 32
    def __init__(self, pretrained: bool = False,
                 is_bias: bool = True,
                 is_bias_FRN: bool = True,
                 dim_desc: int = 128,
                 drop_rate: float = 0.3,
                 eps_l2_norm: float = 1e-10):
        super().__init__()
        self.eps_l2_norm = eps_l2_norm
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FilterResponseNorm2d(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FilterResponseNorm2d(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FilterResponseNorm2d(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

        self.desc_norm = nn.LocalResponseNorm(2 * self.dim_desc, 2.0 * self.dim_desc, 0.5, 0.0)
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls['liberty'], map_location=lambda storage, loc: storage
            )
            self.load_state_dict(pretrained_dict, strict=True)
        self.eval()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.desc_norm(x + self.eps_l2_norm)
        x = x.view(x.size(0), -1)
        return x
    
    
# interface for pySLAM
class HyNetFeature2D: 
    def __init__(self, do_cuda=True):    
        print('Using HyNetFeature2D') 
        self.do_cuda = do_cuda & torch.cuda.is_available()
        print('cuda:',self.do_cuda)     
        device = torch.device("cuda:0" if self.do_cuda else "cpu")        
                
        torch.set_grad_enabled(False)
                        
        # mag_factor is how many times the original keypoint scale
        # is enlarged to generate a patch from a keypoint        
        self.mag_factor = 12.0
        
        # inference batch size        
        self.batch_size = 1024 
        self.process_all = False # process all the patches at once           
        
        print('==> Loading pre-trained network.')        
        self.model = HyNet(pretrained=True)
        if self.do_cuda:
            self.model.cuda()
            print('Extracting on GPU')
        else:
            print('Extracting on CPU')
            self.model.cpu()        
        self.model.eval()            
        print('==> Successfully loaded pre-trained network.')            
            
    
    def compute_des_batches(self, patches):
        n_batches = int(len(patches) / self.batch_size) + 1
        descriptors_for_net = np.zeros((len(patches), 128))
        for i in range(0, len(patches), self.batch_size):
            data_a = patches[i: i + self.batch_size, :, :, :].astype(np.float32)
            data_a = torch.from_numpy(data_a)
            if self.do_cuda:
                data_a = data_a.cuda()
            data_a = Variable(data_a)
            # compute output
            with torch.no_grad():
                out_a = self.model(data_a)
            descriptors_for_net[i: i + self.batch_size,:] = out_a.data.cpu().numpy().reshape(-1, 128) 
        return descriptors_for_net    
    
        
    def compute_des(self, patches):                  
        patches = torch.from_numpy(patches).float()
        patches = torch.unsqueeze(patches,1)
        descrs = []
        for P in range(0, patches.shape[0], 1024):
          if self.do_cuda:
            p = patches[P:P+1024].cuda()
          with torch.no_grad():            
              descrs.append(self.model(p).detach().cpu().numpy().reshape(-1, 128)) 
        descrs = np.concatenate(descrs, axis=0)
        return descrs
                   
                  
    def compute(self, img, kps, mask=None):  #mask is a fake input  
        num_kps = len(kps)
        des = []            
        if num_kps>0:
            if not self.process_all: 
                # compute descriptor for each patch 
                patches = extract_patches_tensor(img, kps, patch_size=32, mag_factor=self.mag_factor) 
                # patches /= 255.
                # patches -= 0.443728476019
                # patches /= 0.20197947209        
                patches = (patches/255. - 0.443728476019)/0.20197947209                      
                des = self.compute_des_batches(patches).astype(np.float32) 
            else: 
                # compute descriptor by feeeding the full patch tensor to the network  
                t = time.time()
                if False: 
                    # use python code 
                    patches = extract_patches_array(img, kps, patch_size=32, mag_factor=self.mag_factor)
                else:
                    # use faster cpp code 
                    patches = extract_patches_array_cpp(img, kps, patch_size=32, mag_factor=self.mag_factor)
                patches = np.asarray(patches)
                if kVerbose:
                    print('patches.shape:',patches.shape)                
                # patches /= 255.
                # patches -= 0.443728476019
                # patches /= 0.20197947209                     
                patches = (patches/255. - 0.443728476019)/0.20197947209     
                if kVerbose:                         
                    print('patch elapsed: ', time.time()-t)
                des = self.compute_des(patches)
        if kVerbose:
            print('descriptor: UAVPatches, #features: ', len(kps), ', frame res: ', img.shape[0:2])                  
        return kps, des
