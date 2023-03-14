#!pip3 install torch==1.5.0 !pip3 install torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html 
import torch

from torch import nn
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from torch.nn import Dropout
from torch.nn import Sigmoid
from torch.nn import BatchNorm2d,AvgPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import ConvTranspose2d
from torch.nn import Softmax  
from torch.nn import AvgPool2d
from torch.nn import BatchNorm2d

import torch.optim as optim
from torch.utils.data import TensorDataset
from .layers.Lambda import Lambda
from .utils.torchsummary import summary as _summary
from .utils.History import History
from .utils.Regularizer import Regularizer
from .layers.Layer_01 import Layer_01 
 
from .Convolution_uno_01   import conv_layer_universal_uno_04,conv_layer_universal_uno_05
import numpy as np
from enum import Enum


class conv_layer_downsample_01(Layer_01):
    """
    Конволюционный слой даунсемплинга при кодировании тензора
    Подкласс Layer_01 служит для определение лосс, алгоритма сходимости,
    регуляризаций, вывода саммари архитектуры, 
    загрузка-сохранение весов, History лосса при обучении 
    """
    def __init__(self, numfilters1_in,   numfilters1_out,  bias_, L1 = 0., L2 = 0.,device = None):
        """
        На вход подаются следующие параметры:
         :param numfilters1_in - размер входного вектора
         :param numfilters1_out - размер выходного вектора
         :param bias_ - смещение применяется к конволюционному слою Conv2d
         :param L1  и L2 - параметры для регуляризации
         :param device - Выбор CPU/GPU(:cuda)
        """
        super(conv_layer_downsample_01, self).__init__()
        
        self.class_name = self.__class__.__name__
        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        #Конволюционный слой
        _layer_conv_31 = Conv2d(numfilters1_in,numfilters1_out, kernel_size=(3, 3),
                            stride=(1, 1), padding = (1, 1), padding_mode = 'zeros', bias = bias_)
        
         
        #Ускорение с помощью БатчНорм
        _layer_batch_norm_1 = BatchNorm2d(num_features=numfilters1_out)#, affine=False)
         
        #двумерное усреднение с окном (2, 2)
        _layer_pooling_1 = AvgPool2d(kernel_size=(2, 2))     
        # Функция активации ЛикиРелу       
        _layer_activation_1 = LeakyReLU(0.05) 
        
        self.add_module('conv_31', _layer_conv_31)
         
        self.add_module('batch_norm_1', _layer_batch_norm_1)
         
        self.add_module('pooling_1', _layer_pooling_1)
        self.add_module('activation_1', _layer_activation_1)
        
        self.to(self.device)
        self.reset_parameters()
        
         
    def forward(self,img_23_32_64_32 ):
         
        img_31 = self._call_simple_layer('conv_31', img_23_32_64_32)
        img_32 = self._call_simple_layer('batch_norm_1', img_31)
        img_33 = self._call_simple_layer('activation_1', img_32)
        img_34_16_32_64 = self._call_simple_layer('pooling_1', img_33)

        

        return img_34_16_32_64
    
class Layer_06(torch.nn.Module):
    """
    общий шаблон класса нейросеть-пайторч с базовыми опциями, таких как:
    определение лосс, алгоритма сходимости, регуляризаций, вывода саммари архитектуры, 
    загрузка-сохранение весов, History лосса при обучении  
    класс опирается на стандартные модули utils и layers 
    """
    def __init__(self, *input_shapes , **kwargs):
        super(Layer_06, self).__init__(**kwargs )
        self.input_shapes = input_shapes
        self.eps_=10**(-20)
        self._criterion = None
        self._optimizer = None
    
    # Обнуление параметров
    def reset_parameters(self):
        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            return (-lim, lim)
        
        for module in self._modules.values():
            if hasattr(module, 'weight') and (module.weight is not None):
                module.weight.data.uniform_(*hidden_init(module))
            if hasattr(module, 'bias') and (module.bias is not None):
                module.bias.data.fill_(0)

    # регуляризатор. Либо Адам либо SGD
    def _get_regularizer(self):
        raise Exception("Need to override method _get_regularizer()!");
        
    # Функция для визуального построения слоев сетки
    def summary(self):
        _summary(self, input_size = self.input_shapes, device = self.device)

    def weights_is_nan(self):
        is_nan = False
        for module in self._modules.values():
            if hasattr(module, 'weight'):
                if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                     (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                    is_nan = True
                    break
            if hasattr(module, 'bias'):
                if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                    is_nan = True
                    break
            
        return is_nan

    # Сохранение весов
    def save_state(self, file_path):
        torch.save(self.state_dict(), file_path)
    
    # Загрузка весов
    def load_state(self, file_path):
        try:
            print()
            print('Loading preset weights... ', end='')

            self.load_state_dict(torch.load(file_path))
            self.eval()
            is_nan = False
            for module in self._modules.values():
                if hasattr(module, 'weight'):
                    if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                         (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                        is_nan = True
                        break
                if hasattr(module, 'bias'):
                    if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                        is_nan = True
                        break
                    
            if (is_nan):
                raise Exception("[Error]: Parameters of layers is NAN!")
                
            print("Ok.")
        except Exception as e:
            print("Fail! ", end='')
            print(str(e))
            print("[Action]: Reseting to random values!")
            self.reset_parameters()
    
    # Функция потерь кросс-энтропия
    def cross_entropy_00(self, pred, soft_targets):
        return -torch.log(self.eps_+torch.mean(torch.sum(soft_targets * pred, -1)))
    
    # Функция потерь MSE
    def MSE_00(self, pred, soft_targets):
        return   torch.mean(torch.mean((soft_targets - pred)**2, -1)) 

    # Функция для сбора параметров перед обучением сети. В эту функцию вводим функцию потерь, оптимизатор, также скорость обучение.
    def compile(self, criterion, optimizer,   **kwargs):
        
        if criterion == 'mse-mean':
            self._criterion = nn.MSELoss(reduction='mean')
            self.flag0986556=0
            
        elif criterion == 'mse-sum':
            self._criterion = nn.MSELoss(reduction='sum')
            self.flag0986556=0
            
        elif criterion == '000':
            self._criterion = self.MSE_00 
        elif criterion == '001':
            self._criterion = self.cross_entropy_00
        elif criterion == 'torch_cross':
            self._criterion = nn.CrossEntropyLoss()
            self.flag0986556=1

        else:
            raise Exception("Unknown loss-function!")
            
        if (optimizer == 'sgd'):
             
            momentum = 0.2
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            if ('momentum' in kwargs.keys()):
                momentum = kwargs['momentum']
            self._optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum);
        elif (optimizer == 'adam'):
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            self._optimizer   = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False) 
             
        else:
            raise Exception("Unknown optimizer!")
            
 
    def _call_simple_layer(self, name_layer, x):
        y = self._modules[name_layer](x)
        if self.device.type == 'cuda' and not y.is_contiguous():
            y = y.contiguous()
        return y
    
    def _contiguous(self, x):
        if self.device.type == 'cuda' and not x.is_contiguous():
            x = x.contiguous()
        return x


class atrous_pyramid_00(Layer_01):
    """
    Специальный класс разреженная пирамида из диплаба
    Подкласс Layer_01 служит для определение лосс, алгоритма сходимости,
    регуляризаций, вывода саммари архитектуры, 
    загрузка-сохранение весов, History лосса при обучении 
    """
    def __init__(self,num_filteres_in=3, num_filteres_out=1,num_filteres_middle=1, L1 = 0., L2 = 0.,device = None,show=0):
        """
        На вход подаются следующие параметры:
         :param num_filteres_in - размер канальности тензора на вход
         :param num_filteres_out - размер канальности тензора на выход
         :param num_filteres_middle - промежуточный размер канальности тензора
         :param L1 и L2 - параметры для регуляризации
         :param device - Выбор CPU/GPU(:cuda)
        """
        super( atrous_pyramid_00 , self).__init__()         
        self.class_name = self.__class__.__name__        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L1=L1
        self.L2=L2
        self.show=show
        self.regularizer = Regularizer(L1, L2)
        # Импортируем слой conv_layer_universal_uno_05. Он состоит из конволюции+БатчНорм+активация по выбору
        # Цифры после активации. Первая цифра - Размер окна(kernel size), вторая и третья цифры это dilation - расстояние между элементами ядра
        self.add_module('conv_layer_universal_uno_05_00',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,2,1,self.device  
                    ))
        self.add_module('conv_layer_universal_uno_05_01',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,4,2,self.device                                                               
                                                                                 ))
        self.add_module('conv_layer_universal_uno_05_02',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,6,3,self.device                                                               
                                                                                 ))
        self.add_module('conv_layer_universal_uno_05_03',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,8,4,self.device 
            ))
        p_z=0 
        k_z=1
        bias_=1
        _layer_conv_31 = Conv2d(4*num_filteres_middle, 8*num_filteres_middle, kernel_size=(k_z, k_z),
                                stride=(1, 1), padding = (p_z, p_z), padding_mode = 'zeros', bias=bias_)
        self.add_module('conv_31', _layer_conv_31)     
        _layer_activation_31 = LeakyReLU(0.05)  
        self.add_module('activation_31', _layer_activation_31)
        _layer_conv_41 = Conv2d(8*num_filteres_middle, num_filteres_out, kernel_size=(k_z, k_z),
                                stride=(1, 1), padding = (p_z, p_z), padding_mode = 'zeros', bias=bias_)
        
        self.add_module('conv_41', _layer_conv_41) 
        _layer_batch_norm_3 = BatchNorm2d(num_filteres_out)
        self.add_module('batch_norm_3', _layer_batch_norm_3)
        _layer_activation_41 = LeakyReLU(0.05)  
        self.add_module('activation_41', _layer_activation_41)

        
        
        self.to(self.device)
        self.reset_parameters()
      
    def forward(self, ref0):
         
         
         
        # четыре раза прогоняется полученный вектор через слои описанные в Ините. Затем они мерджится между собой
        q00= self._modules['conv_layer_universal_uno_05_00'](ref0)
        q01= self._modules['conv_layer_universal_uno_05_01'](ref0)
        q02= self._modules['conv_layer_universal_uno_05_02'](ref0)
        q03= self._modules['conv_layer_universal_uno_05_03'](ref0)
        merged_00 = torch.cat((q00, q01, q02, q03 ),axis=1)
        if self.show:
            print('merged_00',merged_00.shape)
        q04= self._modules['conv_31'](merged_00) # Конволюционный слой
        q05= self._modules['activation_31'](q04) # Активация ЛикиРелу
        q06= self._modules['conv_41'](q05)       # Конволюционный слой
        q07= self._modules['batch_norm_3'](q06)  # БатчНормализация      
        q08= self._modules['activation_41'](q07) # Активация ЛикиРелу
        if self.show:
            print('q05 ',q05.shape)
            print('q08 ',q08.shape)
        
        y = self._contiguous(q08)
        return y

class conv_layer_universal_upsample_00(Layer_06):
    """
    Специальный деконволюционный слой для апсемплинга тензора при декодировании
    Подкласс Layer_06 служит для определение лосс, алгоритма сходимости,
    регуляризаций, вывода саммари архитектуры, 
    загрузка-сохранение весов, History лосса при обучении 
    """    
    def __init__(self, numfilters_in, numfilters_out, k_size, bias_, L1 = 0., L2 = 0., device = None):
        """
        На вход подаются следующие параметры:
         :param num_filteres_in - размер канальности тензора на вход
         :param num_filteres_out - размер канальности тензора на выход
         :param k_size - Размер окна
         :param bias_ - смещение применяется к конволюционному слою Conv2d
         :param L1 и L2 - параметры для регуляризации
         :param device - Выбор CPU/GPU(:cuda)
        """        
        super(conv_layer_universal_upsample_00, self).__init__()
        
        self.class_name = self.__class__.__name__
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_deconv_01 =  ConvTranspose2d(numfilters_in,numfilters_out, kernel_size=(k_size, k_size), 
                                stride=(2, 2), padding = (1, 1),padding_mode = 'zeros', output_padding=(1, 1), bias = True)
        self.add_module('deconv_01', _layer_deconv_01 )
        _layer_activation_D0 = ReLU() 
        self.add_module('activation_D0', _layer_activation_D0) 
        _layer_conv_4 = Conv2d(numfilters_out, numfilters_out, kernel_size=(k_size, k_size),
                     stride=(1, 1), padding=(1, 1), padding_mode = 'zeros', bias = True)
        self.add_module('conv_4', _layer_conv_4 )
        _layer_batch_norm_1 = BatchNorm2d(numfilters_out)
        self.add_module('batch_norm_1', _layer_batch_norm_1)
        _layer_activation_D1 = ReLU() 
        self.add_module('activation_D1', _layer_activation_D1) 
        
    def forward(self, img_23_32_64_32):
        img_31 = self._modules['deconv_01'](img_23_32_64_32) # Деконволюционный слой
        img_32 = self._modules['activation_D0'](img_31)      # Активация Релу
        img_33 = self._modules['conv_4'](img_32)             # Обычный ковлюционный слой
        img_33 = self._modules['batch_norm_1'](img_33)       # Батч Нормализация
        img_34 = self._modules['activation_D1'](img_33)      # Активация Релу
        
        return img_34

class fun_of_factor_00(Layer_01):
    """
    Модуль полносвязной сетки, состоящий из линейных слоев, функций активаций и ДропАутов.
    Подкласс Layer_01 служит для определение лосс, алгоритма сходимости,
    регуляризаций, вывода саммари архитектуры, 
    загрузка-сохранение весов, History лосса при обучении 
    """   
    def __init__(self, Size_,z1,z2,z3,z4,z5,  last_activate,  device = None):
        """
        На вход подаются следующие параметры:
         :param Size_ - Исходный размер
         :param z1,z2,z3,z4,z5 - Размеры на разных линейный слоях. Могут быть и входныи и выходным размером
         :param last_activate - Выбор какая функция активация применяется в конце
         :param device - Выбор CPU/GPU(:cuda)
        """          
        super(fun_of_factor_00, self).__init__()
        self.Size = Size_ 
        self.last_activate=last_activate
        self.class_name = self.__class__.__name__
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
         
        _layer_D00 = Linear( self.Size,  self.Size, bias = True)
        self.add_module('D00', _layer_D00) 
        _layer_Dropout00 = Dropout(0.1)
        self.add_module('Dropout00', _layer_Dropout00) 
        
        _layer_activation_LW1 = LeakyReLU(0.05) 
        self.add_module('activation_LW1', _layer_activation_LW1) 
        
        _layer_D01 = Linear(self.Size, z1, bias = True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.1)
        self.add_module('Dropout01', _layer_Dropout01) 
        
        
        
        _layer_activation_LW2 = LeakyReLU(0.05) 
        self.add_module('activation_LW2', _layer_activation_LW2) 
        
        _layer_D02 = Linear(z1, z2, bias = True)
        self.add_module('D02', _layer_D02)

        _layer_activation_LW3 = LeakyReLU(0.05) 
        self.add_module('activation_LW3', _layer_activation_LW3) 
        
        _layer_D03 = Linear(z2,z3, bias = True)
        self.add_module('D03', _layer_D03)
        _layer_activation_LW4 = LeakyReLU(0.05) 
        self.add_module('activation_LW4', _layer_activation_LW4) 

        
        
        _layer_D04 = Linear(z3,z4, bias = True)
        self.add_module('D04', _layer_D04)
        _layer_activation_LW5 = LeakyReLU(0.05) 
        self.add_module('activation_LW5', _layer_activation_LW5) 
        _layer_D05 = Linear(z4,z5, bias = True)
        self.add_module('D05', _layer_D05)
        
        
        if last_activate == 'sigmoid':
            _layer_activation_D6 = Sigmoid()
            self.add_module('activation_D6', _layer_activation_D6)
        
    def forward(self, x):
         
         
        
        l1=self._modules['D00'](x)  # Линейный слой с исходным размером на вход и выход
        dense_01=self._modules['activation_LW1'](l1) # ЛикиРелу активация
        l2=self._modules['D01'](dense_01) # Линейный слой на вход Size_ на выход размер z1
        
        if 1:
            dense_02=self._modules['activation_LW2'](l2) # ЛикиРелу активация
            l3=self._modules['D02'](dense_02) # Линейный слой на вход z1 на выход размер z2
            dense_03=self._modules['activation_LW3'](l3) # ЛикиРелу активация
            l4=self._modules['D03'](dense_03) # Линейный слой на вход z2 на выход размер z3
            dense_04=self._modules['activation_LW4'](l4) # ЛикиРелу активация
            l5=self._modules['D04'](dense_04) # Линейный слой на вход z3 на выход размер z4
            dense_05=self._modules['activation_LW5'](l5) # ЛикиРелу активация
            l6=self._modules['D05'](dense_05) # Линейный слой на вход z4 на выход размер z5
         

        if self.last_activate == 'sigmoid': # Если включена функция активации, то выбираем сигмоида
            l7 = self._modules['activation_D6'](l6)
        y=l6    
        y = self._contiguous(y)
        return y

class fully_conn_layer_universal_00(Layer_01):
    """
    Полносвязный линейный слой, состоящий из линейный преобразований, функций активаций и ДропАутов
    Подкласс Layer_01 служит для определение лосс, алгоритма сходимости,
    регуляризаций, вывода саммари архитектуры, 
    загрузка-сохранение весов, History лосса при обучении 
    """       
    def __init__(self, Size_, last_activate, device = None):
        """
        На вход подаются следующие параметры:
         :param Size_ - Список с числами, которые используются как входы или выходы в различных линейных слоях.
         :param last_activate - Выбор какая функция активация применяется в конце
         :param device - Выбор CPU/GPU(:cuda)
        """          
        super(fully_conn_layer_universal_00, self).__init__()
        self.Size = Size_[0]
        self.last_activate=last_activate
        self.class_name = self.__class__.__name__
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_D00 = Linear( self.Size,  Size_[1], bias = True)
        self.add_module('D00', _layer_D00) 
        _layer_Dropout00 = Dropout(0.2)
        self.add_module('Dropout00', _layer_Dropout00) 
        
        _layer_activation_LW1 = LeakyReLU(0.2) 
        self.add_module('activation_LW1', _layer_activation_LW1) 
        
        _layer_D01 = Linear(Size_[1], Size_[2], bias = True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.2)
        self.add_module('Dropout01', _layer_Dropout01) 
        
        
        
        _layer_activation_LW2 = LeakyReLU(0.05) 
        self.add_module('activation_LW2', _layer_activation_LW2) 
        _layer_Dropout02 = Dropout(0.1)
        self.add_module('Dropout02', _layer_Dropout02) 
        
        _layer_D02 = Linear(Size_[2], Size_[3], bias = True)
        self.add_module('D02', _layer_D02)

        _layer_activation_LW3 = LeakyReLU(0.05) 
        self.add_module('activation_LW3', _layer_activation_LW3) 
        
        _layer_D03 = Linear(Size_[3], Size_[4], bias = True)
        self.add_module('D03', _layer_D03)
        
        _layer_activation_LW4 = LeakyReLU(0.05) 
        self.add_module('activation_LW4', _layer_activation_LW4) 
        
        _layer_D04 = Linear(Size_[4], Size_[-1], bias = True)
        self.add_module('D04', _layer_D04)
        
        
        
        if last_activate == 'sigmoid':
            _layer_activation_D4 = Sigmoid()
            self.add_module('activation_last', _layer_activation_D4)
        self.to(self.device)
        self.reset_parameters()
        
    def forward(self, x):
         
         
        
        l1=self._modules['D00'](x) # Линейный слой на вход size[0] на выход размер size[1]
        l1=self._modules['Dropout00'](l1) # ДропАут с вероятность 0.2. Необходи для обнуления некоторых нейронов
        dense_01=self._modules['activation_LW1'](l1) # ЛикиРелу активация
        l2=self._modules['D01'](dense_01) # Линейный слой на вход size[1] на выход размер size[2]
        l2=self._modules['Dropout01'](l2) # ДропАут с вероятность 0.2. Необходи для обнуления некоторых нейронов
        dense_02=self._modules['activation_LW2'](l2) # ЛикиРелу активация
        l3=self._modules['D02'](dense_02) # Линейный слой на вход size[2] на выход размер size[3]
        l3=self._modules['Dropout02'](l3) # ДропАут с вероятность 0.1. Необходи для обнуления некоторых нейронов
        dense_03=self._modules['activation_LW3'](l3) # ЛикиРелу активация
        l4=self._modules['D03'](dense_03) # Линейный слой на вход size[3] на выход размер size[4]
        dense_04=self._modules['activation_LW4'](l4) # ЛикиРелу активация
        l5=self._modules['D04'](dense_04) # Линейный слой на вход size[4] на выход размер size[5]
        if self.last_activate == 'sigmoid': # Если в ласт_активейшен написано сигмоид, тогда в виде последней активации используется сигмоида
            l5 = self._modules['activation_last'](l5) 
             
        y=l5    
        y = self._contiguous(y)
        return y

class segmentat_001_wire(Layer_06):
    """
    Основная модель сегментации изображения с попиксельной классификацией сегментов. прототип- диплаб
    Подкласс Layer_06 служит для определение лосс, алгоритма сходимости,
    регуляризаций, вывода саммари архитектуры, 
    загрузка-сохранение весов, History лосса при обучении 
    """         
    def __init__(self, imageSize,  last_activate, L1 = 0., L2 = 0., device = None, numclasses=10 ):
        """
        На вход подаются следующие параметры:
         :param imageSize - Размер картинки
         :param last_activate - Выбор какая функция активация применяется в конце
         :param L1 и L2 - параметры для регуляризации
         :param numclasses - количество выходных нейроннов в конволюционном слое
         :param device - Выбор CPU/GPU(:cuda)
        """           
        super(segmentat_001_wire, self).__init__( (imageSize[0],imageSize[1],3)   )    

        self.class_name = self.__class__.__name__
        self.last_activate = last_activate
        self.cannal_in= imageSize[2]
        self.numclasses= numclasses
        self.imageSize = imageSize
        self.regularizer = Regularizer(L1, L2)
        self.show=0 
        self.L1=L1
        self.L2=L2
        self.X_int_level=torch.zeros(1,1,4*imageSize[0],4*imageSize[1])
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Вручную заданные параметры для слоя пирамиды
        num_filteres_out=64
        num_filteres_middle=32
        num_filteres_in=124
        
        self.atrous_pyramid_00 = atrous_pyramid_00\
        (num_filteres_in , num_filteres_out ,num_filteres_middle, self.L1, self.L2, self.device,self.show)
       
        self.atrous_pyramid_01 = atrous_pyramid_00\
        (3 , 24 ,12, self.L1, self.L2, self.device,self.show)
        self.atrous_pyramid_02 = atrous_pyramid_00\
        (42 , 64 ,24, self.L1, self.L2, self.device,self.show)

        # Полносвязаный линейный слой с уменьшением размерности. Ласт активация сигмоид. Есть Линейные преобзраования+ДропАут+функции активации
        self.add_module('fully_conn_layer_universal_00',\
                        fully_conn_layer_universal_00([624,128,128,64,64,64],'sigmoid', self.device))

        # Полносвязаный линейный слой с уменьшением размерности. Ласт активация сигмоид. Есть только Линейные преобразования+функции активанции
        self.add_module('fun_of_factor_00',\
                        fun_of_factor_00(32,24,12,12,12,3,'sigmoid', self.device)) 
        
        # По флагу не используется
        if 0:
            self.im_to_32_features_00=im_to_32_features_01(L1, L2, self.device)
            self.sketch_to_64_features_00=sketch_to_64_features_01(L1, L2, self.device)

        # Четыре конволюционных слоя с увелечением колличества каналов изображения
        self.conv_layer_universal_01_downsampl=conv_layer_downsample_01(24, 32,    True, self.L1, self.L2, self.device ) 
        self.conv_layer_universal_02_downsampl=conv_layer_downsample_01(32, 42,    True, self.L1, self.L2, self.device )
        self.conv_layer_universal_03_downsampl=conv_layer_downsample_01(64, 86,    True, self.L1, self.L2, self.device ) 
        self.conv_layer_universal_04_downsampl=conv_layer_downsample_01(86, 124,   True, self.L1, self.L2, self.device ) 

        # Четыре ДеКонволюционных слоя с уменьшением каналов изображения
        self.conv_layer_universal_01_upsampl=conv_layer_universal_upsample_00(64, 48, 3,  True, self.L1, self.L2, self.device )    
        self.conv_layer_universal_02_upsampl=conv_layer_universal_upsample_00(48, 32, 3,  True, self.L1, self.L2, self.device )     
        self.conv_layer_universal_03_upsampl=conv_layer_universal_upsample_00(32+42, 32, 3,  True, self.L1, self.L2, self.device )    
        self.conv_layer_universal_04_upsampl=conv_layer_universal_upsample_00(32, 24, 3,  True, self.L1, self.L2, self.device )     
          
         
        # Конволюционный слой из Конвалюции+БатчНормализации+Функция активации. 
        self.add_module('conv_layer_universal_uno_1', conv_layer_universal_uno_04(24+24, 24, 1,'relu',5,2, self.device))
        self.add_module('conv_layer_universal_uno_2', conv_layer_universal_uno_04(24, 24, 1,'relu',1,0, self.device))
        self.add_module('conv_layer_universal_uno_3', conv_layer_universal_uno_04(24, self.numclasses, 1,'linear',1,0, self.device))

        # Функция активации СофтМакс
        _layer_SfTMax = Softmax(dim = -1)
        self.add_module('SfTMax', _layer_SfTMax) 
        # Функция активации Сигмоида
        _layer_Sgmd = Sigmoid()
        self.add_module('Sgmd', _layer_Sgmd)  

        # Деконволюционный слой с уменьшением канальности изображения
        self.encoder_00_upsampl=conv_layer_universal_upsample_00(128, 64, 3,  True, self.L1, self.L2, self.device )  

        # Конволюционный слой с увелечением колличества каналов изображения
        _encoder_map=conv_layer_downsample_01(64, 128,    True, self.L1, self.L2, self.device ) 
        self.add_module('encoder_map',_encoder_map) 

        # Перемешивание вдоль каналов в разреженных пирамидах
        _layer_conv_D0 = Conv2d(128, 150, kernel_size=(1, 1),
                                stride=(1, 1),   padding_mode = 'zeros', bias=True)
        _layer_conv_D1 = Conv2d(150, 150, kernel_size=(1, 1),
                                stride=(1, 1),   padding_mode = 'zeros', bias=True)
        _layer_conv_D2 = Conv2d(150, 128, kernel_size=(1, 1),
                                stride=(1, 1),   padding_mode = 'zeros', bias=True)

        self.add_module('D00',_layer_conv_D0)
        self.add_module('D01',_layer_conv_D1)
        self.add_module('D02',_layer_conv_D2)
        
        # Функция активации ЛикиРелу
        _layer_activation_LeakyReLU = LeakyReLU(0.05)  
        self.add_module('LeakyReLU',_layer_activation_LeakyReLU)
 
        self.to(self.device)
        
        self.reset_parameters()
    #####################################################

    def forward(self, im_wire):
	# применение модели. на вход- 3 канальный имедж от 0 до 1 стандартного размера, задаваемое параметрами класса
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
         
           
        x_input = (  im_wire,im_wire)
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

         
        im_wire = self._contiguous(_x_input[0] )
         
         
        ##############
        
         
        # Смена канальности изображения
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
 
        ref0  = _layer_permut_channelfirst(im_wire)

        if self.show:
            print('ref0',ref0.shape)
        
        im_0=self.atrous_pyramid_01(ref0) # Пирамида, размерность (вход = 3, промежуточный = 12, выход = 24)
        if self.show:
            print('im_0',im_0.shape)

        im_01_dwnsmpl=self.conv_layer_universal_01_downsampl(im_0) # Конволюция с увелечением размера канальности изображения
        if self.show:
            print('im_01_dwnsmpl',im_01_dwnsmpl.shape)

        im_02_dwnsmpl=self.conv_layer_universal_02_downsampl(im_01_dwnsmpl) # Конволюция 2 с увелечением размера канальности изображения
        if self.show:
            print('im_02_dwnsmpl',im_02_dwnsmpl.shape)
            
        im_03= self.atrous_pyramid_02(im_02_dwnsmpl) # Пирамида, размерность (вход = 42, промежуточный = 24, выход = 64)
        if self.show:
            print('im_03',im_03.shape)

        im_03_dwnsmpl=self.conv_layer_universal_03_downsampl(im_03) # Конволюция 3 с увелечением размера канальности изображения
        if self.show:
            print('im_03_dwnsmpl',im_03_dwnsmpl.shape)

        im_04_dwnsmpl=self.conv_layer_universal_04_downsampl(im_03_dwnsmpl) # Конволюция 4 с увелечением размера канальности изображения
        if self.show:
            print('im_04_dwnsmpl',im_04_dwnsmpl.shape)
        ##############################

        atrous_pyramid_= self.atrous_pyramid_00(im_04_dwnsmpl) # Пирамида, размерность (вход = 124, промежуточный = 32, выход = 64)
        if self.show:
            print('atrous_pyramid_ ',atrous_pyramid_.shape)
         
         
        atrous_pyramid_00=self.encoder_map(atrous_pyramid_) # Конволюционный слой с увелечением колличества каналов изображения
        
        # Перемешивание вдоль каналов в разреженных пирамидах c активацией Ликирелу
        atrous_pyramid_1=self.LeakyReLU(self.D00(atrous_pyramid_00))
        atrous_pyramid_2=self.LeakyReLU(self.D01(atrous_pyramid_1))
        atrous_pyramid_3=self.LeakyReLU(self.D02(atrous_pyramid_2))

        # Деконволюционный слой с уменьшением канальности изображения
        init_encoder=self.encoder_00_upsampl(atrous_pyramid_3) 
        if self.show:
            print('atrous_pyramid_00 ',atrous_pyramid_00.shape)
            print('atrous_pyramid_1 ',atrous_pyramid_1.shape)
            print('atrous_pyramid_2 ',atrous_pyramid_2.shape)
            print('atrous_pyramid_3 ',atrous_pyramid_3.shape)
            print('init_encoder',init_encoder.shape)
            
        
        # Деконволюционный слой с уменьшением канальности изображения. Вход 64, выход 48
        im_01_upsmpl=self.conv_layer_universal_01_upsampl(init_encoder)
        if self.show:
            print('im_01_upsmpl',im_01_upsmpl.shape)

        # Деконволюционный слой с уменьшением канальности изображения. Вход 48, выход 32
        im_02_upsmpl=self.conv_layer_universal_02_upsampl(im_01_upsmpl)
        if self.show:
            print('im_02_upsmpl',im_02_upsmpl.shape)
            print('im_02_dwnsmp',im_02_dwnsmpl.shape)

        #Конкатенируем полученный вектор выше im_02_upsmpl и im_02_dwnsmpl
        merged_01 = torch.cat((im_02_upsmpl,im_02_dwnsmpl ),axis=1) 
        
        # Деконволюционный слой с уменьшением канальности изображения. Вход 74, выход 32
        im_03_upsmpl=self.conv_layer_universal_03_upsampl(merged_01)
        if self.show:
            print('im_03_upsmpl',im_03_upsmpl.shape)

        # Деконволюционный слой с уменьшением канальности изображения. Вход 32, выход 24
        im_04_upsmpl=self.conv_layer_universal_04_upsampl(im_03_upsmpl)
        if self.show:
            print('im_04_upsmpl',im_04_upsmpl.shape)

        #Конкатенируем полученный вектор выше im_04_upsmpl и im_0
        merged_02 = torch.cat((im_04_upsmpl, im_0),axis=1)
        if self.show:
            print(' merged_02', merged_02.shape)

        # Конвлюционный слой. Вход 48 выход 24 + Релу активация
        im_05=self.conv_layer_universal_uno_1(merged_02)
        if self.show:
            print('im_05',im_05.shape)

        # Конвлюционный слой. Вход 24 выход 24 + Релу активация
        im_06=self.conv_layer_universal_uno_2(im_05)
        if self.show:
            print('im_06',im_06.shape)

        # Конвлюционный слой. Вход 24 выход 10.
        im_07=self.conv_layer_universal_uno_3(im_06)
        if self.show:
            print('im_07',im_07.shape)
        
        # Меняем каналы
        q01=_layer_permut_channellast(im_07)

        #По флагу выбирается функции активации СофтМакс
        if 1:
            q02=self.SfTMax(q01)
        else:
            q02=self.Sgmd(q01)
        if self.show:
            print('q02 ',q02.shape)
        x = q02
        x = self._contiguous(x)

        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x
    
    # Применяется оптимизатор. В модели Адам
    def _get_regularizer(self):
        return self.regularizer
    
    # Фит -  Используется для обучения модели. На вход даталоадер и колличество эпох
    # Для Треин используется loss_batch_02, для валидации loss_batch_00
    def fit_dataloader_00(self, loader,  epochs = 1, validation_loader = None):
        #обучение сети на базе даталоадера
        
        if (self._criterion is None): 
            raise Exception("Loss-function is not select!")
        # Из функции compile получаем функцию потерь = mse-mean

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")
        # Из функции compile получаем оптимизатор = Адам

        # Функция Потерь. Сам критерий получаем из compile.
        # xb, yb - Получаем из Даталоадера      
        def loss_batch_00(loss_func, xb, yb, opt=None):
             

            pred = self(*xb)
             
            if isinstance(pred, tuple):
                pred0 = pred[0]
                del pred
            else:
                pred0 = pred

             

            loss = loss_func(pred0, yb)
            acc=-1

            # Считаем точность по флагу
            if 1:
                _, predicted = torch.max(pred.data, dim = -1)
                _, ind_target = torch.max(yb, dim = -1)
                correct = (predicted == ind_target).sum().item()
                
                 
                
                acc = correct / (len(yb)*self.imageSize[0]*self.imageSize[1])
                 
            if 0:
                print('loss = loss_func(pred0, yb)')
                print(pred0.shape)
                print(yb.shape)
             
            del pred0
               
             
            _regularizer = self._get_regularizer()
            
            reg_loss = 0
            for param in self.parameters():
                reg_loss += _regularizer(param)
                
            loss += reg_loss
             
                      
            if count_%5==0:
                print("*", end='')
        
            loss_item = loss.item()
            
            del loss
            del reg_loss
             
            return loss_item, len(yb) , acc
              

        def loss_batch_02(loss_func, xb, yb, opt=None):
        # оптимизация весов для текуего батча на базе лосс   
             
             
            pred = self(xb)
             
            if isinstance(pred, tuple):
                pred0 = pred[0]
                del pred
            else:
                pred0 = pred

            if self.flag0986556 == 1:
                 
                yb = torch.argmax(yb, dim=-1)
                yb = torch.reshape(yb, (-1,))
                pred0 = torch.reshape(pred0, (pred0.shape[1] * pred0.shape[2], pred0.shape[3]))
             

            loss = loss_func(pred0, yb)
            
            acc=-1
            if 0:
                _, predicted = torch.max(pred.data, dim = -1)
                _, ind_target = torch.max(yb, dim = -1)
                correct = (predicted == ind_target).sum().item()
                
                 
                
                acc = correct / (len(yb)*self.imageSize[0]*self.imageSize[1])
                 
            if 0:
                print('loss = loss_func(pred0, yb)')
                print(pred0.shape)
                print(yb.shape)
        
             
            del pred0
               
             
            _regularizer = self._get_regularizer()
            
            reg_loss = 0
            for param in self.parameters():
                reg_loss += _regularizer(param)
                
            loss += reg_loss
             
            if opt is not None:
                with torch.no_grad():
                      
                    opt.zero_grad()
                     
                    loss.backward()
                     
                    opt.step()
                     
            if count_ % 5 == 0:
                print("*", end='')
        
            loss_item = loss.item()
            
            del loss
            del reg_loss
             
            return loss_item, len(yb) , acc    
              
        ###############################      

        history = History()
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
            

            
            
            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            accs=[]
            count_=0
            
             
 
            for s in loader:
                 
                train_ds = TensorDataset( torch.FloatTensor(s['Anchor'].numpy()).to(self.device),\
                                         
                                        torch.FloatTensor(s['label'].numpy()).to(self.device))
                                       
                  
                
                
                 
                im_to_segm=train_ds.tensors[0] # Получаем Anchor
                 
                target=train_ds.tensors[1] # Получаем label

                # Все загружаем в функцию loss_batch
                losses_, nums_ ,acc_  =   loss_batch_02(self._criterion, \
                                                   ( im_to_segm ),\
                                                   target,  self._optimizer)                                                                                                       
                                 
                count_+=1                                     
                losses.append(losses_)
                nums.append(nums_ )
                accs.append(acc_)
                
                
            print("]", end='')

            # Считаем Потери и точность модели.
            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            acc = np.sum(np.multiply(accs, nums)) / sum_nums
             
            ### test mode ###
            if validation_loader is not None:

                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                accs=[]
                for s in validation_loader:
                #s = next(iter(loader))
                 
                    val_ds = TensorDataset( torch.FloatTensor(s['Anchor'].numpy()).to(self.device),\
                                             
                                            torch.FloatTensor(s['label'].numpy()).to(self.device))





                    im_to_segm=val_ds.tensors[0]# 
                     
                    target=val_ds.tensors[1]

                     

                    losses_, nums_ ,acc_  =   loss_batch_00(self._criterion, \
                                                       ( im_to_segm ),\
                                                       target,  self._optimizer)                                                                                                       
                      
                    
                                                                                                                         

                                                   
                    losses.append(losses_)
                    nums.append(nums_ )
                    accs.append(acc_)
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                val_acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'acc': acc, 'val_loss': val_loss,'val_acc ':val_acc })
                print(' - Loss: {:.6f}, Accuracy: {:.6f}'.format(loss, acc), end='')
                 
                print(' - Test-loss: {:.6f}, Test-accuracy: {:.6f}'.format(val_loss,val_acc), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
##############################################################
################################################################3
def kernel_Ntxtr(k):
    a=np.eye((k**2))
    b=np.reshape(a,[-1,1,k,k])
    return b
def int_log_2(x):
    return  int(np.log(x ) /np.log(2 ) )


####################################################################################################
class im2txtrTnzr04(Layer_06):
    def __init__(self, param, device=None):
         
        super( im2txtrTnzr04, self).__init__(  param["imageSize"]) 
        modules = []
        
        self.param = param
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
        self.input_shape = tuple(param["imageSize"])
        self.dilation = tuple(param["dilation"])
        self.size_layers = tuple(param["w_"])
        for w_ , dl_z in zip(self.size_layers,self.dilation):
            W5 =  kernel_Ntxtr(w_)
            
            step_=1#max(1,int(w_**2/500))
            W5=W5[::step_,...]
            if 0:
                print(W5.shape)
                ai_2(W5[0,0,:,:])
                ai_2(W5[1,0,:,:])
                ai_2(W5[2,0,:,:])
                ai_2(W5[3,0,:,:])
                ai_2(W5[w_,0,:,:])
                ai_2(W5[2*w_,0,:,:])
                ai_2(W5[3*w_,0,:,:])
                ai_2(W5[3*w_+6,0,:,:])
                ai_2(W5[5*w_+3,0,:,:])
                 

                print('===================================')
            out_channels_=W5.shape[0]
            strides_=2**int_log_2(w_)
            layer_conv_64 = torch.nn.Conv2d(in_channels=self.input_shape[-1], out_channels=w_**2, kernel_size=(w_,w_),\
                                        stride =(strides_,strides_),\
                                            dilation = (dl_z,dl_z), \
                                            #padding = (dl_z*int(w_/2) -dl_z , dl_z*int(w_/2)-dl_z ),\
                                            padding_mode = 'zeros', bias=0)
            
            layer_conv_64_T = torch.nn.ConvTranspose2d(in_channels=w_**2,out_channels= self.input_shape[-1], kernel_size=(w_,w_),\
                                                         stride =(strides_,strides_) ,\
                                                            padding = (0, 0),\
                                                            output_padding=(0, 0),\
                                                            padding_mode = 'zeros', bias=0)
             
             
            layer_conv_64.weight.data = torch.FloatTensor(W5)   
            layer_conv_64_T.weight.data = torch.FloatTensor(W5)  
            self.add_module('conv_'+str(w_)+'_'+str(dl_z), layer_conv_64) 
            self.add_module('conv_T_'+str(w_)+'_'+str(dl_z), layer_conv_64_T) 
 
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))         
        self.to(self.device)
        
    def get_features(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
        x = self._layer_permut_channelfirst(x)
        features = {}
        for w_ , dl_z in zip(self.size_layers,self.dilation):
                    
            x_=self._modules['conv_'+str(w_)+'_'+str(dl_z)] (x)
            features[str(w_)+'_'+str(dl_z)]=x_
        
        
         
 
        return features
    def represent_features(self,features ):
         
        for w_ , dl_z in zip(self.size_layers,self.dilation):
                    
             
            print( features[str(w_)+'_'+str(dl_z)].shape )
            q=np.mean(features[str(w_)+'_'+str(dl_z)].detach().numpy(),1)[0,...]
             
            
            ai_2(q)
        
        
         
 
        return 1



    def get_features_01(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
       
        features = {}
        for w_ , dl_z in zip(self.size_layers,self.dilation):
                    
            x_=self._modules['conv_'+str(w_)+'_'+str(dl_z)] (x)
            features[str(w_)+'_'+str(dl_z)]=x_
        
        
         
 
        return features

    def get_features_02(self, x, name_of_w):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
       
         
        
                    
            
        x_=self._modules['conv_T_'+name_of_w] (x)
             
        
        
         
 
        return x_

    def map_00(self,txtr_,routine):
        a, b, c, d = txtr_.size()  # a=batch size(=1)        
        features =txtr_.reshape([a * b, c * d])
        G = torch.mm(features, features.t())  # compute the gram product


        if routine:
            (U, S, V)=torch.pca_lowrank(G, q=10, center=0, niter=2)
            map_2=torch.mm(U, U.t())
        else:
            map_2=G.div(a * b * c * d)
        return map_2
 
    def map_24(self,x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
        x = self._layer_permut_channelfirst(x)
        map_ = {}
        for w_ , dl_z in zip(self.size_layers,self.dilation):
             
            x24=self._modules['conv_'+str(w_)+'_'+str(dl_z)] (x)
            map_0=self.map_00(x24,1)
            map_[str(w_)+'_'+str(dl_z)]=map_0
         
         
        
         
        return map_
    
    
    def forward(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        
        x = self.conv_64_1(x)
        print(x.shape)
        if (to_numpy):
            if (self.device  == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()

        if (to_list):
            if (self.device  == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        return x
    
    def summary(self):
        _summary(self, input_size = self.input_shape, device = self.device)
        
    def ONNXexport(self, filename, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            x = torch.Floa
            tTensor(x).to(self.device)
        torch.onnx.export(self, x, filename, export_params=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
#########################################################
