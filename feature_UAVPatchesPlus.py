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


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        #self.features.apply(weights_init)

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)
    
    
# interface for pySLAM
class UAVPatchesPlusFeature2D: 
    def __init__(self, do_cuda=True):    
        print('Using UAVPatchesPlusFeature2D')         
        self.model_base_path = config.cfg.root_folder + '/thirdparty/hardnet/'        
        self.model_weights_path = './Pretrained/UAVPatches+.pth'
        #print('model_weights_path:',self.model_weights_path)
        
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
        self.model = HardNet()
        self.checkpoint = torch.load(self.model_weights_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(self.checkpoint['state_dict'])
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
            print('descriptor: UAVPatchesPlus, #features: ', len(kps), ', frame res: ', img.shape[0:2])                  
        return kps, des
    
