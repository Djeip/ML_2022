# -*- coding: utf-8 -*-
"""
@author: user
"""

import torch
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from torch.nn import BatchNorm2d
from torch.nn import Sigmoid
from . import layers

class conv_layer_universal_uno_02(layers.Layer):
    def __init__(self, numfilters_in, numfilters_out, bias_, device = None):
        super(conv_layer_universal_uno_02, self).__init__()
        
        self.class_name = self.__class__.__name__
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_conv_31 = Conv2d(numfilters_in, numfilters_out, kernel_size=(3, 3),
                                stride=(1, 1), padding = (1, 1), padding_mode = 'zeros', bias=bias_)
        
        _layer_batch_norm_1 = BatchNorm2d(numfilters_out)
        _layer_activation_1 = LeakyReLU(0.05)

        self.add_module('conv_31', _layer_conv_31)
        self.add_module('batch_norm_1', _layer_batch_norm_1)
        self.add_module('activation_1', _layer_activation_1)
        
    def forward(self, img_23_32_64_32):
        img_31 = self._modules['conv_31'](img_23_32_64_32)
        img_32 = self._modules['batch_norm_1'](img_31)
        img_33 = self._modules['activation_1'](img_32)
        
        return img_33
    ##################################################
class conv_layer_universal_uno_03(layers.Layer):
    def __init__(self, numfilters_in, numfilters_out, bias_,last_activate,k_z, device = None):
        super(conv_layer_universal_uno_03, self).__init__()
        
        self.class_name = self.__class__.__name__
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_conv_31 = Conv2d(numfilters_in, numfilters_out, kernel_size=(k_z, k_z),
                                stride=(1, 1), padding = (1, 1), padding_mode = 'same', bias=bias_)
        
        _layer_batch_norm_1 = BatchNorm2d(numfilters_out)
        

        self.add_module('conv_31', _layer_conv_31)
        self.add_module('batch_norm_1', _layer_batch_norm_1)
        self.last_activate=last_activate
        if last_activate == 'sigmoid':
            _layer_activation_D4 = Sigmoid()
            self.add_module('activation_1', _layer_activation_D4)
            
        elif last_activate == 'relu':
            _layer_activation_1 = LeakyReLU(0.05)
            self.add_module('activation_1', _layer_activation_1)
        elif last_activate == 'linear':
             pass
        
        
    def forward(self, img_23_32_64_32):
        img_31 = self._modules['conv_31'](img_23_32_64_32)
        img_32 = self._modules['batch_norm_1'](img_31)
        if self.last_activate == 'linear':
            img_33 = img_32
        else:
            img_33 = self._modules['activation_1'](img_32)
        
        return img_33
    
class conv_layer_universal_uno_04(layers.Layer):
    def __init__(self, numfilters_in, numfilters_out, bias_,last_activate,k_z,p_z, device = None):
        super(conv_layer_universal_uno_04, self).__init__()
        
        self.class_name = self.__class__.__name__
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_conv_31 = Conv2d(numfilters_in, numfilters_out, kernel_size=(k_z, k_z),
                                stride=(1, 1), padding = (p_z, p_z), padding_mode = 'zeros', bias=bias_)
        
        _layer_batch_norm_1 = BatchNorm2d(numfilters_out)
        

        self.add_module('conv_31', _layer_conv_31)
        self.add_module('batch_norm_1', _layer_batch_norm_1)
        self.last_activate=last_activate
        if last_activate == 'sigmoid':
            _layer_activation_D4 = Sigmoid()
            self.add_module('activation_1', _layer_activation_D4)
            
        elif last_activate == 'relu':
            _layer_activation_1 = LeakyReLU(0.05)
            self.add_module('activation_1', _layer_activation_1)
        elif last_activate == 'linear':
             pass
        
        
    def forward(self, img_23_32_64_32):
        img_31 = self._modules['conv_31'](img_23_32_64_32)
        img_32 = self._modules['batch_norm_1'](img_31)
        if self.last_activate == 'linear':
            img_33 = img_32
        else:
            img_33 = self._modules['activation_1'](img_32)
        
        return img_33
    
class conv_layer_universal_uno_05(layers.Layer):
    def __init__(self, numfilters_in, numfilters_out, bias_,last_activate,k_z,p_z,dl_z, device = None):
        super(conv_layer_universal_uno_05, self).__init__()
        
        self.class_name = self.__class__.__name__
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_conv_31 = Conv2d(numfilters_in, numfilters_out, kernel_size=(k_z, k_z),
                                stride=(1, 1), padding = (p_z, p_z),dilation = (dl_z,dl_z), padding_mode = 'zeros', bias=bias_)
        
        _layer_batch_norm_1 = BatchNorm2d(numfilters_out)
        

        self.add_module('conv_31', _layer_conv_31)
        self.add_module('batch_norm_1', _layer_batch_norm_1)
        self.last_activate=last_activate
        if last_activate == 'sigmoid':
            _layer_activation_D4 = Sigmoid()
            self.add_module('activation_1', _layer_activation_D4)
            
        elif last_activate == 'relu':
            _layer_activation_1 = LeakyReLU(0.05)
            self.add_module('activation_1', _layer_activation_1)
        elif last_activate == 'linear':
             pass
        
        
    def forward(self, img_23_32_64_32):
        img_31 = self._modules['conv_31'](img_23_32_64_32)
        img_32 = self._modules['batch_norm_1'](img_31)
        if self.last_activate == 'linear':
            img_33 = img_32
        else:
            img_33 = self._modules['activation_1'](img_32)
        
        return img_33
