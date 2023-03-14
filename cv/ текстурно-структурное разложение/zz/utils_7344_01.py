import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mat
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import random
import pandas as pd
import scipy
import scipy.signal
import scipy.io as io
import os
import sys
from scipy import ndimage
from pprint import pprint
import numpy.ma
from PIL import Image, ImageDraw
import cv2
import time
from keras.utils import np_utils 
from skimage.util import random_noise
from typing import List, Tuple, Set, Union, Any
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
import scipy.ndimage.filters as nd_filters
from skimage import measure, color
from scipy.optimize import curve_fit
from statistics import mean
import scipy.ndimage.filters as nd_filters
#########################################################################################
def rand():
    return np.random.rand()
def randn():
    return np.random.randn()

def get_unic_canal(x,cannal_):
    return torch.unsqueeze(x[:,:,: ,cannal_ ], 3)

########################################################
def zeros_list(n):
    elements = [ c for c in range(n)]
    init_M=[elements.copy() for c in range( n)]
    for uyio in range(n):
        for ere in range(n):
            init_M[uyio][ere]=[]

    return init_M
def list_elop_to_confmart_0(init_M ):
    n=len(init_M )
    conf_matrix=np.zeros((n,n))
    for uyio in range(n):
        for ere in range(n):
            if len(init_M[uyio][ere])>0:
                conf_matrix[uyio,ere]=np.sum((np.array(init_M[uyio][ere])>IOU_thresh).astype('uint8')) 
    return conf_matrix.astype('uint8')

def softmax_00(w, t = 1.0):
    e = np.exp(np.array(abs(w)) / t)
    dist = e / np.sum(e)
    return dist
def to_categorical(factor_k,k):
    return np_utils.to_categorical(factor_k, num_classes=k)  
def un_categorical_axis_1(Y_23423):
    return  np.expand_dims(np.argmax(Y_23423, axis = 1),1) 
##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def downscale_2( img):
        img = img.astype(np.uint16)
        img = img[:, 0::2] + img[:, 1::2]
        img = img[0::2, :] + img[1::2, :]
        img >>= 2
        return img.astype(np.float)    
def file2tenzor(pimage):
         
        stream = open(pimage, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        try: 
            bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
            if( bw_.shape[0]==  256 ) and (bw_.shape[1]==256) :

                resize_image =  downscale_2(bw_)
                #resize_image =  bw_ 
            else:
                resize_image = cv2.resize(bw_, (128,128), interpolation = cv2.INTER_AREA)


            return resize_image/255

        except:
            print(pimage)
            return None

class Caption_Class_01:

    def __init__(self, path, flag,scale_):
        #path - путь для доступа к входным данным: для видео - путь к файлу, для папки с файлами - путь к папке, для rtsp/ftp - ссылка
        #flag - Тип входных данных: 0 - видео, 1 - папка с файлами, 2 - rtsp-поток, 3 - ftp-поток (не реализовано)
        if flag==2:
            self.cap = cv2.VideoCapture(path)
        else:
            if flag==1:
                self.files = os.listdir(path)
                #print(self.files)
            elif flag==0:
                self.cap = cv2.VideoCapture(path)
        self.path=path
        self.file=''
        self.input_type=flag
        self.count_file=0
        self.scale_=scale_

    def isOpened(self):
        if self.input_type==1:
            if (self.count_file<len(self.files)):
                return True
            else:
                return False
        else:
            return self.cap.isOpened()

    def read(self):
        if self.input_type==1:
            file_=self.files[self.count_file]
            q_=-1
            open_cv_image=None
            if file_.endswith(".jpg") or file_.endswith(".png") or file_.endswith(".jpeg"):
                self.file=file_
                pil_image = Image.open(self.path+file_).convert('RGB')
                open_cv_image = np.array(pil_image)
                # Convert RGB to BGR
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                q_=1
            self.count_file+=1    
            return q_, open_cv_image
        else:
            return self.cap.read()
        
    def read_01(self ):
        
        if self.scale_ == -1:
            
            if self.input_type==1:
                file_=self.files[self.count_file]
                q_=-1
                open_cv_image=None
                if file_.endswith(".jpg") or file_.endswith(".png") or file_.endswith(".jpeg"):
                    self.file=file_
                    pil_image = Image.open(self.path+file_).convert('RGB')
                    open_cv_image = np.array(pil_image)
                    # print(open_cv_image.shape)
                    # Convert RGB to BGR
                    if open_cv_image.ndim >2:
                        open_cv_image = open_cv_image[:, :, ::-1].copy()
                    else:
                        A_0=np.tile(np.expand_dims(open_cv_image,2),(1,1,3))
                        open_cv_image = A_0
                        if 0:
                            plt.figure(figsize=(5,5))
                            plt.imshow(open_cv_image)
                            plt.show()

                    q_=1
                self.count_file+=1    
                return q_, open_cv_image
            else:
                return self.cap.read()
             
        else:        
            if self.input_type==1:
                file_=self.files[self.count_file]
                q_=-1
                open_cv_image=None
                if file_.endswith(".jpg") or file_.endswith(".png") or file_.endswith(".jpeg"):
                    self.file=file_
                    pil_image = Image.open(self.path+file_)
                    open_cv_image = np.array(pil_image.resize( self.scale_))
                    #print(open_cv_image.shape)
                    #print(open_cv_image.ndim)
 
                    # Convert RGB to BGR
                    if open_cv_image.ndim >2:
                        open_cv_image = open_cv_image[:, :, ::-1].copy()
                    else:
                        A_0=np.tile(np.expand_dims(open_cv_image,2),(1,1,3))
                        open_cv_image = A_0
                        if 0:
                            plt.figure(figsize=(5,5))
                            plt.imshow(open_cv_image)
                            plt.show()
                    q_=1
                self.count_file+=1    
                return q_, open_cv_image
            else:
                ret, fr=self.cap.read() 
                if ret>0:
                    fr1=np.array(Image.fromarray(fr).resize(self.scale_), dtype = 'uint8')
                    return ret, fr1
                else:
                    return -1, 0
 
    def release(self):
        if self.input_type==1:
            return 0
        else:
            return self.cap.release()
        
###################################################################       
def random_warp(frame1,scale_0,scale_1,scale_2,scale_):
     
    W_=frame1.shape[0]
    H_=frame1.shape[1]
    pt=np.array([H_/2+np.random.randint(-scale_1,scale_1) , W_/2+np.random.randint(-scale_1,scale_1)] , dtype='float32')
    w=50
    h=50
    src2 =  np.array([pt+[-w , h ],pt+[-w ,-h  ],pt+[ w ,-h ] ,pt+[ w , h  ]], dtype = "float32")

    dst = np.random.randint(-scale_0,scale_0)+np.array([
        pt+[-w , h ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt+[-w ,-h  ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt+[ w ,-h ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt+[ w , h  ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)]
        ], dtype = "float32")
    width = int(frame1.shape[1] * scale_2)
    height = int(frame1.shape[0] * scale_2)
    dim = (width, height)   
    M = cv2.getPerspectiveTransform(src2,dst)
    warped = cv2.warpPerspective(frame1, M, dim)
    return  warped
   
def random_warp_01(frame1,scale_0,scale_1,scale_2,scale_,w,h):
     
    W_=frame1.shape[0]
    H_=frame1.shape[1]
    pt=np.array([H_/2+np.random.randint(-scale_1,scale_1) , W_/2+np.random.randint(-scale_1,scale_1)] , dtype='float32')
     
    src2 =  np.array([pt+[-w , h ],pt+[-w ,-h  ],pt+[ w ,-h ] ,pt+[ w , h  ]], dtype = "float32")

    dst = np.random.randint(-scale_0,scale_0)+np.array([
        pt+[-w , h ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt+[-w ,-h  ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt+[ w ,-h ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt+[ w , h  ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)]
        ], dtype = "float32")
    width = int(frame1.shape[1] * scale_2)
    height = int(frame1.shape[0] * scale_2)
    dim = (width, height)   
    M = cv2.getPerspectiveTransform(src2,dst)
    warped = cv2.warpPerspective(frame1, M, dim)
    return  warped
###################################################################    
###########################################  
def random_warp_02(frame1,scale_0,scale_1,scale_2,scale_,shift_,w,h):
     
    W_=frame1.shape[0]
    H_=frame1.shape[1]
    pt=np.array([H_/2+np.random.randint(-scale_1,scale_1) , W_/2+np.random.randint(-scale_1,scale_1)] , dtype='float32')
    pt1=pt +[np.random.randint(-shift_,shift_),np.random.randint(-shift_,shift_)]
    src2 =  np.array([pt+[-w , h ],pt+[-w ,-h  ],pt+[ w ,-h ] ,pt+[ w , h  ]], dtype = "float32")

    dst = np.random.randint(-scale_0,scale_0)+np.array([
        pt1+[-w , h ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt1+[-w ,-h  ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt1+[ w ,-h ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)],
        pt1+[ w , h  ] +[np.random.randint(-scale_,scale_),np.random.randint(-scale_,scale_)]
        ], dtype = "float32")
    width = int(frame1.shape[1] * scale_2)
    height = int(frame1.shape[0] * scale_2)
    dim = (width, height)   
    M = cv2.getPerspectiveTransform(src2,dst)
    warped = cv2.warpPerspective(frame1, M, dim)
    return  warped
################################################3

        
def plot_im_2(img,img2):
     
    plt.figure(figsize=(25,10))

    plt.plot(img.ravel() ,'k')
    plt.plot(img2.ravel() ,'r')
     
    plt.show()  
def plot_im_3(img,img2,img3):
     
    plt.figure(figsize=(10,5))

    plt.plot(img.ravel() ,'y')
    plt.plot(img2.ravel() ,'k')
    plt.plot(img3.ravel() ,'r')
     
    plt.show()  
        

        
##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
####   sift mahal utils  #####################################################3
##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def sift_extract_shifts_00(matches,kp2,kp1):
    shifts=[]
    for i in range(len(matches)):
        shifts.append(np.array(kp2[matches[i].trainIdx].pt)-np.array(kp1[matches[i].queryIdx].pt))
         
     
    return np.array(shifts) 

def index_mahal_reg_00(shifts,rho):
    m_1=abs(mahal(shifts , shifts) )
    ii_=np.where(m_1<1)[0]

    m_2=abs(mahal(shifts , shifts[ii_,:]) )
    ii_1=np.where(m_2<rho)[0]
    plot_im_2(m_1,m_2)
    return ii_1
def index_mahal_reg_01(shifts,rho):
    m_1=abs(mahal(shifts , shifts) )
    ii_=np.where(m_1<1)[0]

    m_2=abs(mahal(shifts , shifts[ii_,:]) )
    ii_1=np.where(m_2<rho)[0]
    #plot_im_2(m_1,m_2)
    return ii_1

def find_anomaly_00(hst_class,ll, final_anomaly_level,show):
    # погружаемся в диффузную карту

    hst_dff= np.array(diffusion_representation_helek3( zentrir_str(hst_class),0,2,'sc',0)[0]).T

    # maхаланобис-дистанции в диффузной карте
    try:
        m_1=abs(mahal(hst_dff , hst_dff) )
        print(m_1)
    except Exception:
            m_1=np.ones(hst_dff.shape[0],1)
    # анализируя махаланобис- расстояния, определяем аномальные точки
    m_2=m_1
    for i in range(7): 
        anomaly_level=3*median_001(m_1)
        ii= mat.find(m_1<anomaly_level)
        if len(ii)<15 : # если аномалий слишком много то бессмысленно выделять отдельный класс
            break
        #ii= mat.find(m_1<10)
        m_2=abs(mahal(hst_dff , hst_dff[ii,:]) ) 
        if show:
            #plt.plot(m_0.T ,color = 'blue', linestyle = 'dashed', label = 'mahal origin')
            plt.plot(  m_1.T,color='r', label='mahal dff')
            plt.plot(  m_2.T,color='c', label='mahal dff processed')
            plt.plot( anomaly_level+0*m_2.T,color='b', label='anomaly level')
            plt.legend(loc = 'upper left')
            plt.show()
        m_1=m_2
    for i in range(0): 
        #ii= mat.find(m_1<4*median_001(m_1))
        ii= mat.find(m_1<10)
        m_2=abs(mahal(hst_dff , hst_dff[ii,:]) ) 
        if show:

            plt.plot(  m_1.T,color='r', label='mahal dff')
            plt.plot(  m_2.T,color='c', label='mahal dff processed')
            plt.plot(  3*median_001(m_1)+0*m_2.T,color='b', label='anomaly level')
            plt.legend(loc = 'upper left')
            plt.show()
        m_1=m_2

    anomaly_level=final_anomaly_level
    ii_anomaly= mat.find(m_2>anomaly_level) 

    num=ll # максимальное число аномалий
    l=len(ii_anomaly)
    if l>num:
        idx_ = np.random.permutation(l)
        idx = to_1D(idx_[0:min(num, l)])
        ii_anomaly=ii_anomaly[idx]
        ii_reg= np.array(list(set(range(hst_dff.shape[0])) - set(ii_anomaly)))
    else:
        ii_reg= mat.find(m_2<anomaly_level)
    return ii_anomaly,ii_reg

def simultanious_sift_Ne_collection_00(matches2,kp1,kp2,sketch1,sketch2,Mask,w_,thr_similarity_sift,show,M1,M2):

    X_u0=[]
    X_u1=[]
    for ind_match in range(len(matches2)):

        beg_0=int( kp2[matches2[ind_match].trainIdx].pt[1])
        beg_1=int( kp2[matches2[ind_match].trainIdx].pt[0])
        if Mask[beg_0,beg_1]>0:
            U_sketch1=sketch2[beg_0:beg_0+w_,beg_1:beg_1+w_]
            seq_vect_1=np.reshape(U_sketch1,[1,w_**2])
            if show:
                U_M2=M2[beg_0:beg_0+w_,beg_1:beg_1+w_,0]

            
            beg_0=int( kp1[matches2[ind_match].queryIdx].pt[1])
            beg_1=int( kp1[matches2[ind_match].queryIdx].pt[0])
            if Mask[beg_0,beg_1]>0:
                if matches2[ind_match].distance<thr_similarity_sift:
                    X_u0.append(seq_vect_1)

                    U_sketch2=sketch1[beg_0:beg_0+w_,beg_1:beg_1+w_]
                    seq_vect_2=np.reshape(U_sketch2,[1,w_**2])
                    X_u1.append(seq_vect_2)
                    
                    if show:
                        
                        U_M1=M1[beg_0:beg_0+w_,beg_1:beg_1+w_,0]
                        ai_2(np.concatenate([U_sketch1,U_sketch2],1))
                        ai_2(np.concatenate([U_M1,U_M2],1))
                        plot_im_2(seq_vect_1,seq_vect_2)

                        print(np.mean(U_sketch1*U_sketch2))
    X_u1  = np.reshape(np.array(X_u1)   ,[len(X_u1),w_**2]   )    
    X_u0  = np.reshape(np.array(X_u0)   ,[len(X_u1),w_**2]   )    
    Y_=np.ones((X_u0.shape[0],1))
    return X_u1,X_u0,Y_

def simultanious_sift_Ne_collection_01(matches2,kp1,kp2,sketch1,sketch2,Mask,w_,thr_similarity_sift,show,M1,M2):

    X_u0=[]
    X_u1=[]
    Y_=[]
    w_0=int(w_/2)
    for ind_match in range(len(matches2)):

        beg_0=int( kp2[matches2[ind_match].trainIdx].pt[1])
        beg_1=int( kp2[matches2[ind_match].trainIdx].pt[0])
        if Mask[beg_0,beg_1]>0:
            U_sketch1=sketch2[beg_0-w_0:beg_0+w_0,beg_1-w_0:beg_1+w_0]
            seq_vect_1=np.reshape(U_sketch1,[1,w_**2])
            if show:
                U_M2=M2[beg_0-w_0:beg_0+w_0,beg_1-w_0:beg_1+w_0,0]

            
            beg_0=int( kp1[matches2[ind_match].queryIdx].pt[1])
            beg_1=int( kp1[matches2[ind_match].queryIdx].pt[0])
            if Mask[beg_0,beg_1]>0:
                if matches2[ind_match].distance<thr_similarity_sift:
                    X_u0.append(seq_vect_1)

                    U_sketch2=sketch1[beg_0-w_0:beg_0+w_0,beg_1-w_0:beg_1+w_0]
                    seq_vect_2=np.reshape(U_sketch2,[1,w_**2])
                    X_u1.append(seq_vect_2)
                    Y_.append([0.8])
                    if show:
                        
                        U_M1=M1[beg_0-w_0:beg_0+w_0,beg_1-w_0:beg_1+w_0,0]
                        ai_2(np.concatenate([U_sketch1,U_sketch2],1))
                        ai_2(np.concatenate([U_M1,U_M2],1))
                        plot_im_2(seq_vect_1,seq_vect_2)

                        print(np.mean(U_sketch1*U_sketch2))
                    for hjk in range(10):
                        shift_0=int((np.random.rand()-0.5)*3)
                        shift_1=int((np.random.rand()-0.5)*3)
                        mesh_=np.sqrt( shift_0**2+shift_1**2)
                        if mesh_ >0:
                            q_=  0.7/mesh_
                            beg_01=beg_0+shift_0
                            beg_11=beg_1+shift_1
                             
                            U_sketch2=sketch1[beg_01-w_0:beg_01+w_0,beg_11-w_0:beg_11+w_0]
                            seq_vect_2=np.reshape(U_sketch2,[1,w_**2])
                            X_u0.append(seq_vect_1)
                            X_u1.append(seq_vect_2)
                            Y_.append([q_])
                        
    X_u1  = np.reshape(np.array(X_u1)   ,[-1,w_**2]   )    
    X_u0  = np.reshape(np.array(X_u0)   ,[-1,w_**2]   )    
    Y_=np.array(Y_)
    return X_u1,X_u0,Y_


def cross_sift_Ne_collection_00(X_u11,X_u00,kama,show):


    Y_0=[]
    Y_1=[]
    for jhyg in range(kama):
        ivr=int(np.random.rand()*(X_u11.shape[0]-1)) 
        iqqr=int(np.random.rand()*(X_u00.shape[0]-1)) 
        Y_0.append(X_u11[ivr,:]) 
        Y_1.append(X_u00[iqqr,:])
        if show:
            NU=X_u11[ivr,:]
            ai_2(np.reshape(NU,[w_,w_]))
            NU1=X_u00[iqqr,:]
            ai_2(np.reshape(NU1,[w_,w_]))
    Y_0=np.array(Y_0)
    Y_1=np.array(Y_1)
    Z_=np.zeros((Y_0.shape[0],1))
    return Y_0,Y_1,Z_
def simult_shift_collect_01(sketch1,Mask,w_,kama,q_,show):
    X_u0=[]
    X_u1=[]
    for kh in range(kama):
        i_=int(np.random.rand()*sketch1.shape[0])
        j_=int(np.random.rand()*sketch1.shape[1])
        if Mask[i_,j_]>0:
            U_sketch1=sketch1[i_:i_+w_,j_:j_+w_]
            if not (np.sum(U_sketch1+0.5)==0):
                beg_0=i_+int((np.random.rand()-0.5)*q_)
                beg_1=j_+int((np.random.rand()-0.5)*q_)
                U_sketch2=sketch1[beg_0:beg_0+w_,beg_1:beg_1+w_]
                seq_vect_2=np.reshape(U_sketch2,[1,w_**2])
                seq_vect_1=np.reshape(U_sketch1,[1,w_**2])
                X_u1.append(seq_vect_2)
                X_u0.append(seq_vect_1)
                if show:

                    ai_2(np.concatenate([U_sketch1,U_sketch2],1))
    X_u1  = np.reshape(np.array(X_u1)   ,[len(X_u1),w_**2]   )    
    X_u0  = np.reshape(np.array(X_u0)   ,[len(X_u1),w_**2]   )    
    Y_=np.ones((X_u0.shape[0],1)) 
    return X_u1 ,X_u0 ,Y_


def simult_shift_collect(sketch1,Mask,w_,kama,show):
    X_u0=[]
    X_u1=[]
    for kh in range(kama):
        i_=int(np.random.rand()*sketch1.shape[0])
        j_=int(np.random.rand()*sketch1.shape[1])
        if Mask[i_,j_]>0:
            U_sketch1=sketch1[i_:i_+w_,j_:j_+w_]
            beg_0=i_+int((np.random.rand()-0.5)*6)
            beg_1=j_+int((np.random.rand()-0.5)*6)
            U_sketch2=sketch1[beg_0:beg_0+w_,beg_1:beg_1+w_]
            seq_vect_2=np.reshape(U_sketch2,[1,w_**2])
            seq_vect_1=np.reshape(U_sketch1,[1,w_**2])
            X_u1.append(seq_vect_2)
            X_u0.append(seq_vect_1)
            if show:

                ai_2(np.concatenate([U_sketch1,U_sketch2],1))
    X_u1  = np.reshape(np.array(X_u1)   ,[len(X_u1),w_**2]   )    
    X_u0  = np.reshape(np.array(X_u0)   ,[len(X_u1),w_**2]   )    
    Y_=np.ones((X_u0.shape[0],1)) 
    return X_u1 ,X_u0 ,Y_

def simult_corr_collect(sketch1, w_,kama,show):
    X_u0=[]
    X_u1=[]
    Y_=[]

    for kh in range(kama):
        i_0=int(np.random.rand()*(sketch1.shape[0]-w_))
        j_0=int(np.random.rand()*(sketch1.shape[1]-w_))
        U_sketch0=sketch1[i_0:i_0+w_,j_0:j_0+w_]
        if not (np.sum(U_sketch0+0.5)==0):
            i_1=int(np.random.rand()*(sketch1.shape[0]-w_))
            j_1=int(np.random.rand()*(sketch1.shape[1]-w_))
            U_sketch1=sketch1[i_1:i_1+w_,j_1:j_1+w_]
            if not (np.sum(U_sketch1+0.5)==0):


                seq_vect_0=np.reshape(U_sketch0,[1,w_**2])
                seq_vect_1=np.reshape(U_sketch1,[1,w_**2])
                r_=np.corrcoef(np.concatenate([seq_vect_0,seq_vect_1]))[0,1]
                if r_<0.2:
                    X_u1.append(seq_vect_0)
                    X_u0.append(seq_vect_1)
                    Y_.append(0)
                    if 0:

                        ai_2(np.concatenate([U_sketch1,U_sketch0],1))
                        print(np.corrcoef(np.concatenate([seq_vect_0,seq_vect_1]))[0,1])

                if r_>0.6:
                    X_u1.append(seq_vect_0)
                    X_u0.append(seq_vect_1)
                    Y_.append(1)


                    if show:

                        ai_2(np.concatenate([U_sketch1,U_sketch0],1))
                        print(np.corrcoef(np.concatenate([seq_vect_0,seq_vect_1]))[0,1])

    X_u1  = np.reshape(np.array(X_u1)   ,[len(X_u1),w_**2]   )    
    X_u0  = np.reshape(np.array(X_u0)   ,[len(X_u1),w_**2]   )    
    Y_=np.reshape(Y_, [X_u0.shape[0],1]) 
    return X_u1 ,X_u0 ,Y_

def simult_corr_collect_01(sketch1, w_,kama,show):
    X_u0=[]
    X_u1=[]
    Y_=[]

    for kh in range(kama):
        i_0=int(np.random.rand()*(sketch1.shape[0]-w_))
        j_0=int(np.random.rand()*(sketch1.shape[1]-w_))
        U_sketch0=sketch1[i_0:i_0+w_,j_0:j_0+w_]
        if not (np.sum(U_sketch0+0.5)==0):
            i_1=int(np.random.rand()*(sketch1.shape[0]-w_))
            j_1=int(np.random.rand()*(sketch1.shape[1]-w_))
            U_sketch1=sketch1[i_1:i_1+w_,j_1:j_1+w_]
            if not (np.sum(U_sketch1+0.5)==0):


                seq_vect_0=np.reshape(U_sketch0,[1,w_**2])
                seq_vect_1=np.reshape(U_sketch1,[1,w_**2])
                r_=np.corrcoef(np.concatenate([seq_vect_0,seq_vect_1]))[0,1]
                if r_<0.2:
                    X_u1.append(seq_vect_0)
                    X_u0.append(seq_vect_1)
                    Y_.append(0)
                    if 0:

                        ai_2(np.concatenate([U_sketch1,U_sketch0],1))
                        print(np.corrcoef(np.concatenate([seq_vect_0,seq_vect_1]))[0,1])

                if r_>0.6:
                    X_u1.append(seq_vect_0)
                    X_u0.append(seq_vect_1)
                    Y_.append(0.8)


                    if show:

                        ai_2(np.concatenate([U_sketch1,U_sketch0],1))
                        print(np.corrcoef(np.concatenate([seq_vect_0,seq_vect_1]))[0,1])

    X_u1  = np.reshape(np.array(X_u1)   ,[len(X_u1),w_**2]   )    
    X_u0  = np.reshape(np.array(X_u0)   ,[len(X_u1),w_**2]   )    
    Y_=np.reshape(Y_, [X_u0.shape[0],1]) 
    return X_u1 ,X_u0 ,Y_

def sift2mot_00(matches2,kp1,kp2,show):
    points=[]
    velo=[]
    for ind_match in range(len(matches2)):
        beg_00=int( kp2[matches2[ind_match].trainIdx].pt[1])
        beg_01=int( kp2[matches2[ind_match].trainIdx].pt[0])
        beg_10=int( kp1[matches2[ind_match].queryIdx].pt[1])
        beg_11=int( kp1[matches2[ind_match].queryIdx].pt[0])
        v_x=beg_10-beg_00
        v_y=beg_11-beg_01
        points.append([beg_00,beg_01])
        velo.append([v_x,v_y])
        if show:
            print(beg_00,beg_01,v_x,v_y)
    points=np.array(points)
    velo=np.array(velo)

         
    return points,velo

def avrg_mot_compens_00(collect_color):
    mot_compensate=[]
    mot_compensate.append(collect_color[-1])
    for i in range(1,len(collect_bw)):
        flow = cv2.calcOpticalFlowFarneback(collect_bw[-1-i],collect_bw[-1], None, 0.5, 10, 5, 5, 5, 1.5, 0)
        flow[:,:,0] = -flow[:,:,0]+Y_
        flow[:,:,1] = -flow[:,:,1]+X_
        mapped_img = cv2.remap(collect_color[-1-i],flow[:,:,1] ,   flow[:,:,0], cv2.INTER_LINEAR)
        mot_compensate.append(mapped_img)
        #Avrg=np.mean(np.array(mot_compensate),axis=0)
        Avrg=np.mean(np.array(mot_compensate),axis=0)
        return Avrg
    
class KNN_Class_01:

    def __init__(self, points, velo,k_):
        self.trainData = points.astype(np.float32)
        self.responses = (1000+ velo[:,0])*10000+  (1000+ velo[:, 1])#np.random.randint(0,2,(trainData.shape[0],1)).astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.trainData,cv2.ml.ROW_SAMPLE,self.responses)
        self.k_=k_
    def apply_1N(self,newcomer,sigm_):
        ret, results, neighbours, dist = self.knn.findNearest(newcomer.astype(np.float32) , 1)
        neighbours_0=np.concatenate([np.expand_dims(neighbours[:,0]//10000-1000,1),\
                                     np.expand_dims(neighbours[:,0]%10000-1000,1)],1)
        if 0:
            self.Q0=neighbours
            self.Q1=neighbours_0
            print('ret',ret)
            print('neighbours[0]',neighbours[0])
            print('results',results)
            print('neighbours',neighbours)
            print('neighbours_0',neighbours_0)
            print('dist',dist)
        
        Nearest_Neig=np.expand_dims(neighbours_0,2)    
        
        KK=scipy.special.softmax(self.k_*np.exp(-dist/sigm_**2),1)# w of neigh
        Velo_estimated=np.matmul(Nearest_Neig,np.expand_dims(KK,2)) 
        return Nearest_Neig,Velo_estimated
    def apply_2N(self,newcomer,sigm_):
        ret, results, neighbours, dist = self.knn.findNearest(newcomer.astype(np.float32) , 2)
        neighbours_0=np.concatenate([np.expand_dims(neighbours[:,0]//10000-1000,1),np.expand_dims(neighbours[:,0]%10000-1000,1)],1)
        neighbours_1=np.concatenate([np.expand_dims(neighbours[:,1]//10000-1000,1),np.expand_dims(neighbours[:,1]%10000-1000,1)],1)
         
        Nearest_Neig=np.concatenate([np.expand_dims(neighbours_0,2),\
                                     np.expand_dims(neighbours_1,2) ],2)     
        
        KK=scipy.special.softmax(self.k_*np.exp(-dist/sigm_**2),1)# w of neigh
        Velo_estimated=np.matmul(Nearest_Neig,np.expand_dims(KK,2)) 
        return Nearest_Neig,Velo_estimated

    def apply_3N(self,newcomer,sigm_):
        ret, results, neighbours, dist = self.knn.findNearest(newcomer.astype(np.float32) , 3)
        neighbours_0=np.concatenate([np.expand_dims(neighbours[:,0]//10000-1000,1),np.expand_dims(neighbours[:,0]%10000-1000,1)],1)
        neighbours_1=np.concatenate([np.expand_dims(neighbours[:,1]//10000-1000,1),np.expand_dims(neighbours[:,1]%10000-1000,1)],1)
        neighbours_2=np.concatenate([np.expand_dims(neighbours[:,2]//10000-1000,1),np.expand_dims(neighbours[:,2]%10000-1000,1)],1)

        Nearest_Neig=np.concatenate([np.expand_dims(neighbours_0,2),\
                                     np.expand_dims(neighbours_1,2),np.expand_dims(neighbours_2,2)],2)     
        
        KK=scipy.special.softmax(self.k_*np.exp(-dist/sigm_**2),1)# w of neigh
        Velo_estimated=np.matmul(Nearest_Neig,np.expand_dims(KK,2)) 
        
        return Nearest_Neig,Velo_estimated
    def apply_5N(self,newcomer,sigm_):
        ret, results, neighbours, dist = self.knn.findNearest(newcomer.astype(np.float32) , 5)
        neighbours_0=np.concatenate([np.expand_dims(neighbours[:,0]//10000-1000,1),np.expand_dims(neighbours[:,0]%10000-1000,1)],1)
        neighbours_1=np.concatenate([np.expand_dims(neighbours[:,1]//10000-1000,1),np.expand_dims(neighbours[:,1]%10000-1000,1)],1)
        neighbours_2=np.concatenate([np.expand_dims(neighbours[:,2]//10000-1000,1),np.expand_dims(neighbours[:,2]%10000-1000,1)],1)
        neighbours_3=np.concatenate([np.expand_dims(neighbours[:,3]//10000-1000,1),np.expand_dims(neighbours[:,3]%10000-1000,1)],1)
        neighbours_4=np.concatenate([np.expand_dims(neighbours[:,4]//10000-1000,1),np.expand_dims(neighbours[:,4]%10000-1000,1)],1)

        Nearest_Neig=np.concatenate([np.expand_dims(neighbours_0,2),\
                                     np.expand_dims(neighbours_1,2),np.expand_dims(neighbours_2,2),\
                                    np.expand_dims(neighbours_3,2),np.expand_dims(neighbours_4,2)],2)     
        
        KK=scipy.special.softmax(self.k_*np.exp(-dist/sigm_**2),1)# w of neigh
        Velo_estimated=np.matmul(Nearest_Neig,np.expand_dims(KK,2)) 
        
        return Nearest_Neig,Velo_estimated
    def apply_10N(self,newcomer,sigm_):
        ret, results, neighbours, dist = self.knn.findNearest(newcomer.astype(np.float32) , 10)
        neighbours_0=np.concatenate([np.expand_dims(neighbours[:,0]//10000-1000,1),np.expand_dims(neighbours[:,0]%10000-1000,1)],1)
        neighbours_1=np.concatenate([np.expand_dims(neighbours[:,1]//10000-1000,1),np.expand_dims(neighbours[:,1]%10000-1000,1)],1)
        neighbours_2=np.concatenate([np.expand_dims(neighbours[:,2]//10000-1000,1),np.expand_dims(neighbours[:,2]%10000-1000,1)],1)
        neighbours_3=np.concatenate([np.expand_dims(neighbours[:,3]//10000-1000,1),np.expand_dims(neighbours[:,3]%10000-1000,1)],1)
        neighbours_4=np.concatenate([np.expand_dims(neighbours[:,4]//10000-1000,1),np.expand_dims(neighbours[:,4]%10000-1000,1)],1)
        neighbours_5=np.concatenate([np.expand_dims(neighbours[:,5]//10000-1000,1),np.expand_dims(neighbours[:,5]%10000-1000,1)],1)
        neighbours_6=np.concatenate([np.expand_dims(neighbours[:,6]//10000-1000,1),np.expand_dims(neighbours[:,6]%10000-1000,1)],1)
        neighbours_7=np.concatenate([np.expand_dims(neighbours[:,7]//10000-1000,1),np.expand_dims(neighbours[:,7]%10000-1000,1)],1)
        neighbours_8=np.concatenate([np.expand_dims(neighbours[:,8]//10000-1000,1),np.expand_dims(neighbours[:,8]%10000-1000,1)],1)
        neighbours_9=np.concatenate([np.expand_dims(neighbours[:,9]//10000-1000,1),np.expand_dims(neighbours[:,9]%10000-1000,1)],1)

        Nearest_Neig=np.concatenate([np.expand_dims(neighbours_0,2),\
                                    np.expand_dims(neighbours_1,2),np.expand_dims(neighbours_2,2),\
                                    np.expand_dims(neighbours_3,2),np.expand_dims(neighbours_4,2),\
                                    np.expand_dims(neighbours_5,2),np.expand_dims(neighbours_6,2),\
                                    np.expand_dims(neighbours_7,2),np.expand_dims(neighbours_8,2),\
                                    np.expand_dims(neighbours_9,2),\
                                    ],2)     
        
        KK=scipy.special.softmax(self.k_*np.exp(-dist/sigm_**2),1)# w of neigh
        Velo_estimated=np.matmul(Nearest_Neig,np.expand_dims(KK,2)) 
        
        return Nearest_Neig,Velo_estimated
    
def sift2mot_01(matches2,kp1,kp2,show):
    points=[]
    velo=[]
    for ind_match in range(len(matches2)):
        beg_00=int( kp2[matches2[ind_match].trainIdx].pt[1])
        beg_01=int( kp2[matches2[ind_match].trainIdx].pt[0])
        beg_10=int( kp1[matches2[ind_match].queryIdx].pt[1])
        beg_11=int( kp1[matches2[ind_match].queryIdx].pt[0])
        v_x=beg_10-beg_00
        v_y=beg_11-beg_01
        points.append([beg_00,beg_01])
        velo.append([v_x,v_y])
        if show:
            print(beg_00,beg_01,v_x,v_y)
    points=np.array(points)
    velo=np.array(velo)

         
    return points,velo

class Motion_KNN(KNN_Class_01):
    def __init__(self, points, velo,k_,shape_ ,sigm_):
        super(Motion_KNN, self).__init__(points, velo,k_)
        self.sigm_=sigm_
        self.shape_=shape_
        self.xv, self.yv = np.meshgrid(range(shape_[0]), range(shape_[1]), sparse=False, indexing='ij')
    def KNN_motion_00(self ):
        field_xy=np.concatenate([np.reshape(self.xv,[-1,1]),np.reshape(self.yv,[-1,1])],1)
         
        self.mot_xy=np.reshape(self.apply_10N(field_xy,self.sigm_)[1],\
                               [self.shape_[0],self.shape_[1],2])
         
        self.flow=self.mot_xy.copy()
         
        self.flow[:,:,0] =   self.mot_xy[:,:,0]+self.xv 
        self.flow[:,:,1] =   self.mot_xy[:,:,1] +self.yv
    def shift_flow(self,sc_3):
        mapped_  = cv2.remap(sc_3  ,self.flow[:,:,1].astype(np.float32) ,   self.flow[:,:,0].astype(np.float32), cv2.INTER_LINEAR)
        return mapped_ 
    
    def shift_flow_T_00(self,sc_3):
        mapped_  = cv2.remap(sc_3  ,self.flow[:,:,1].astype(np.float32) ,   self.flow[:,:,0].astype(np.float32), cv2.INTER_LINEAR)
        return mapped_ 

######################
#### MORPHOLOGY  #####
#####################
def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = (input_image).copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0),255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out /255.0
def errosia_00(q,anomaly_w,max_):
    fil = np.ones((1, q)) / (q)
    anomaly_w_1 =  scipy.signal.convolve(anomaly_w, fil.ravel(), mode='same') 
    i_0=np.where(np.array(anomaly_w_1)<max_)[0]
    anomaly_w_2=anomaly_w.copy()
    anomaly_w_2[i_0]=0
    return anomaly_w_2
def delation_00(q,anomaly_w,max_):
     
    anomaly_w_1 = errosia_00(q, max_-anomaly_w,max_)
    i_0=np.where(np.array(anomaly_w_1)<max_)[0]
    anomaly_w_2=anomaly_w_1.copy()
    anomaly_w_2[i_0]=0
    anomaly_w_3=max_- anomaly_w_2
    return anomaly_w_3
def morph_00(q,anomaly_w,max_):
    anomaly_w_2= errosia_00(q,anomaly_w,max_)

    return delation_00(q,anomaly_w_2,max_)
def morph_2D_0(img,n):
    kernel = np.ones((n, n), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    erose_img = cv2.erode(dilate_img, kernel, iterations=1)
    return erose_img

def connectivity_components_00(label_gray):
    return measure.label(label_gray, connectivity=2)
    

##################################################################################3
def  rshuk(k,x ,ex):
     
    if(k<0)  :
        y=lshuk(-k,x,ex)
    else  :
        m,n,l=x.shape
        if ex==3  :
            y=np.hstack((x[:,k-1:1-2:-1,:],x[:,1-1:(n-k),:]))
        elif ex==2  :
            y=np.hstack((x[:,k+1-1:2-2:-1,:],x[:,1-1:(n-k),:]))
        elif ex==1  :
            y=np.hstack((x[:,n-k+1-1:n,:],x[:,1-1:(n-k),:]))
        elif ex==7  :
            y=np.hstack((np.zeros((m,k,l)),x[:,1-1:(n-k),:]))
        elif ex==4  :
            y=np.hstack((-x[:,n-k+1-1:n,:],x[:,1-1:(n-k),:]))
        ###
    ###
    return y


def  lshuk(k,x ,ex):

    
    if(k<0)  :
        y=rshuk(-k,x,ex)
    else  :
        m,n,l=x.shape
        if k==0  :
            y=x
        else  :
            if ex==2  :
                y=np.hstack((x[:,k+1-1:n,:],x[:,n-1-1:n-k-2:-1,:]))
            elif ex==3  :
                y=np.hstack((x[:,k+1-1:n,:],x[:,n-1:n-k+1-2:-1,:]))
            elif ex==1  :
                y=np.hstack((x[:,k+1-1:n,:],x[:,1-1:k,:]))
            elif ex==4  :
                y=np.hstack((x[:,k+1-1:n,:],-x[:,1-1:k,:]))
            ###
        ###
    ###
    return y 

def zentrir_str(hst_class):
    return  hst_class - hst_class.mean(axis=1, keepdims=True)
###########################################    
def clear_dir_00(path_out):
    
        try:
            for root, dirs, files in os.walk(path_out, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                    
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        except:
            pass
        
def collect_dir_00(path_in ,path_out):
    
    try:    
        for root, dirs, files in os.walk(path_in, topdown=False):
            for name in files:
                print(os.path.join(root, name))
                print(os.path.join(path_out, name))
                os.replace(os.path.join(root, name), os.path.join(path_out, name))
                #os.remove(os.path.join(root, name))
            for name in dirs:
                    os.rmdir(os.path.join(root, name))


            q=0
    except:
        pass
    
    
def collect_dir_01(path_in ):
    
    collection=[]    
    for root, dirs, files in os.walk(path_in, topdown=False):
        for name in files:
            #print(os.path.join(root, name))
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg"):
             
                collection.append(os.path.join(root, name)) 
            #os.remove(os.path.join(root, name))
        for name in dirs:
            subcollect=collect_dir_01(os.path.join(root, name))
            collection=collection+subcollect 

    return collection
    
##############################################
    
def  sort(x0):

    x=np.mat(x0)
    I = np.argsort(x)
    J=np.array(I).reshape(-1)
    b=x[:, J]
    return b,J


def  sort1(x0,ord_):

    x=np.mat(x0)
    if (ord_>0) :
        I = np.argsort(x,axis=-1, kind='quick')
        JJ=np.array(I).reshape(-1)
        b=x[:, JJ]
    else  :
        if (ord_<0) :
            I = np.argsort(x,axis=-1, kind='quick')
            J=np.array(I).reshape(-1)
            JJ=J[::-1]
            b=x[:, JJ]
    return b,JJ

def median_001(A)  :

    return np.median(np.array(np.asarray(A).ravel())) 

def  sort2(x0,ord_):

    x=np.mat(x0)
    if (ord_>0) :
        I = np.argsort(x,axis=-1, kind='quick')
        JJ=np.array(I).reshape(-1)
        b=x[:, JJ]
    else  :
        if (ord_<0) :
            I = np.argsort(x,axis=-1, kind='quick')
            J=np.array(I).reshape(-1)
            JJ=J[::-1]
            b=x[:, JJ]
    return to_1D(b),JJ


def  sort2D_0(x0,ord_):

    x=np.mat(x0)
    if (ord_>0) :
        I = np.argsort(x,axis=-1, kind='quick')
        x2=np.array([])
        for tgt in range(0, x.shape[0])  :
            J_=I[tgt,:]
            y=x[tgt,J_]
            x2=vstack(x2,y)
    else  :
        I = np.argsort(x,axis=-1, kind='quick')
        x2=np.array([])
        for tgt in range(0, x.shape[0])  :
            J_=I[tgt,:]
            JJ=J_[::-1]
            y=x[tgt,JJ]
            x2=vstack(x2,y)
    return np.mat(x2)
def show_img_1(img,cmap=None):
    plt.figure()
    plt.imshow(img,cmap)
    plt.show()
    
def show_img_10(img,w,h,cmap=None):
    plt.figure(figsize=(w,h))
    plt.imshow(img,cmap)
    plt.show()

def show_img_2(img1, img2, cmap=None):
    plt.figure()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1,cmap)
    ax[1].imshow(img2,cmap)
    plt.show()


def show_img_3(img1, img2, img3, cmap=None):
    plt.figure()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img1,cmap)
    ax[1].imshow(img2,cmap)
    ax[2].imshow(img3,cmap)
    plt.show()
def vizu_file(file_):
         
    stream = open(file_, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
    stream.close()

    bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 

    if( bw_.shape[0]==-256 ) and (bw_.shape[1]==256) :

        resize_image = downscale_2(bw_)
    else:
        resize_image = cv2.resize(bw_, (528,528), interpolation = cv2.INTER_AREA)
    fig = plt.figure(figsize=(15, 15)) 
    plt.axes().set_aspect('equal')
    plt.imshow(resize_image)
    plt.show()

def repr_img_3(a0,a1,a2):
    show_img_1(np.concatenate([  1*(np.expand_dims(a0,2)/np.max(a0)), 0.1*(np.expand_dims(a1,2)/np.max(a1)),\
                   1.5*(np.expand_dims(a1,2)/np.max(a2))],2) )


def max_02(G):

    G=to_1D(G)
    R,I=sort1(G,-1)
    In=I[0]
    Ma=G[In]
    return Ma,In
 

def apply_mask_00(im_2,y1,kk):
    y2=abs(np.power(y1/(y1.max() +0.00000001)   ,kk)) 
    mask1 =  np.maximum(0,np.array(Image.fromarray(y2).resize([im_2.shape[0], im_2.shape[1]], Image.BICUBIC), dtype = 'float'))
    im_3=  np.minimum(255,np.maximum(0,np.multiply (im_2 , numpy.ma.dstack((mask1, mask1, mask1)) )))
    return im_3 
                      


def rand_ones_sparse_00(W3,kk):
    s=(int(np.size(W3, axis=0)),int(np.size(W3, axis=1)))
    VV=np.zeros(s)
    ll_= VV.size
    rr_=np.array([np.random.permutation(ll_)])
    VV=np.reshape(VV, (VV.shape[0] * VV.shape[1]), 'F')
    VV[rr_[0,: np.minimum(kk,ll_)]] = 1
    VV=VV.reshape(int(np.size(W3, axis=0)),int(np.size(W3, axis=1)))
    return VV

def  plt_multi_01_(a,x): #displays graphs by rows of the matrix, does not work for column vectors
    plt.figure(x)
    plt.clf() 
    for uju in range(1,a.shape[0]+1)  :
        plt.plot(to_1D(a[uju-1,:])  ,'y')
    plt.ylabel('the values of the elements')
    plt.xlabel('the indexes of the elements')
    plt.show()
    return 1 
def to_1D(A)  :

    B=A.ravel()
    if  isinstance(A,np.matrix) :
        B=np.array(np.asarray(A).ravel())
    return B

def  eye(size_):

    a=np.size(size_)
    if (a==1) :
        return eye(size_)
    else  :
        return np.eye(size_[1])
def init_cell(n) :
    c=[]
    for uju in range(1,n+1)  :
        c.append([0])
    return c


def fun_hi(a,x) :

    if(x>a) :
        return 0
    else  :
        return 1


def corr(w1,w2) :

    s=np.corrcoef(w1,w2)
    q= s[0,1]
    return q


def eigen_fast(AA,nn):

    m=AA.shape[1-1]
    AA1=AA
    PCA_1=np.array([])
    T_2=np.array([])
    for ioi in range(1,nn+1)  :
        R=np.ones((m,1))
        R=R/np.linalg.norm(R)
        for iui in range(1,50+1)  :
            VR=np.dot(AA1,R)
            R=VR/np.linalg.norm(VR)
        ###
        T2=np.linalg.norm(np.dot(AA1,R))
        PCA_1=hstack(PCA_1,R/np.linalg.norm(R))
        T_2=hstack(T_2,T2)
        II=np.eye(m)-np.dot(R,R.transpose())
        AA1=np.dot(AA1,II)
    ###
    PCA_=PCA_1.transpose()
    T2=T_2
    return PCA_,T2

def projet_to_subsp(E1):

    F_1=np.dot(E1,E1.transpose())
    F_2=quasy_inv_matr_002(np.mat(F_1),0.01)
    L_S=np.dot(F_2,E1)
    Pr_=np.dot(E1.transpose(),L_S)
    return   Pr_


def essential_basis_from_set_04(F,show,qq_):

    E_,ind_=modal_vect_01(F,show)
    n_first=np.sqrt(np.sum(np.power(E_,2)))
    Pr_=projet_to_subsp(np.mat(E_))
    Pr_F=np.dot(F,Pr_)
    F1=F-Pr_F
    for hnm in range(0,F.shape[0])  :
        n_0=np.sqrt(np.sum(np.power(F1,2),1))
        max_no,i_m=max_02(n_0)
        if( max_no/n_first>qq_)  :
            Element_0=np.mat(F)[i_m,:]
            E_=vstack(E_,Element_0)
            ind_=hstack(ind_,i_m)
            Pr_=projet_to_subsp(np.mat(E_))
            Pr_F=np.dot(F,Pr_)
            F1=F-Pr_F
        else  :
            break
        ###
    ###
    return E_,Pr_,ind_ 

###################################################33


def hrd_thr_oper(coeff_09096,hhh):

    ab_c=abs(coeff_09096)
    s_a_c,II_65=sort1(ab_c,-1)
    coeff_0909=1.0*coeff_09096; coeff_0909=to_1D(coeff_0909)
    i1=np.arange(0,np.size(II_65))
    i2=II_65[0:hhh]
    i1_=np.asarray(i1).ravel().tolist()
    i2_=np.asarray(i2).ravel().tolist()
    i3=np.array(list(set(i1_).difference(set(i2_))))
    if len(i3) >0:
        coeff_0909[i3]=0
    return    coeff_0909
####################################################################3

def shift_flow_T_01(I_ref_emb,mot_xy ):

    I_ref_emb_shift=I_ref_emb.copy()
    s_0=I_ref_emb.shape[0]
    s_1=I_ref_emb.shape[1]

    for ikhg in range(s_0):
        for uouih in range(s_1):
            s_x= mot_xy[ikhg,uouih,0] 
            s_y= mot_xy[ikhg,uouih,1] 
            q_x=np.minimum(s_0-1,np.maximum(0,ikhg+int(s_x )))
            q_y=np.minimum(s_1-1,np.maximum(0,uouih+int(s_y )))


            I_ref_emb_shift[ikhg ,uouih,: ]=I_ref_emb[q_x,q_y,:]
    return I_ref_emb_shift

def apply_poly_00(x,coeff):
    Y=0*x+coeff[0]
    for i in range(1,len(coeff)):
         
        Y=Y+coeff[i]*x**i
    return Y
def show_surf_00(xv, yv, f_poly_2d):
    fig = plt.figure(figsize = (7, 5))
    # создаём рисунок пространства с поверхностью
    ax = fig.add_subplot(1, 1, 1, projection = '3d')
    # размечаем границы осей для аргументов
    xval = np.linspace(-1, 1, 1)
    yval = np.linspace(-1, 1, 1)
    surf = ax.plot_surface(
    # отмечаем аргументы и уравнение поверхности
    xv, yv, f_poly_2d, 
    # шаг прорисовки сетки
    # - чем меньше значение, тем плавнее
    # - будет градиент на поверхности
    rstride = 10,
    cstride = 10,
    # цветовая схема plasma
    cmap = cm.plasma)
    plt.show()
    
def random_choose_pair_N_00(B_1scotch,B_0,mot_xy,w_) :
    try:    
        w_0=w_//2
        i_1=w_0+int(np.random.rand()*(B_0.shape[0]-w_))
        j_1=w_0+int(np.random.rand()*(B_0.shape[1]-w_))

        U_sketch1=B_1scotch[i_1-w_0:i_1+w_0,j_1-w_0:j_1+w_0]
        i_2=int(i_1+mot_xy[i_1,j_1,0])
        j_2=int(j_1+mot_xy[i_1,j_1,1])
        U_sketch2=B_0[i_2-w_0:i_2+w_0,j_2-w_0:j_2+w_0]    
        response=np.minimum(np.sum(U_sketch2),np.sum(U_sketch1))-3
    except:
        U_sketch1=0
        U_sketch2=0
        response=-1
    return U_sketch1,U_sketch2,response

def random_choose_pair_N_02(seg_map_1, seg_map,B_1,B_0,mot_xy,w_,l_) :
         
 
    try:    
        w_0=w_//2
        i_1=w_0+int(np.random.rand()*(B_0.shape[0]-w_))
        j_1=w_0+int(np.random.rand()*(B_0.shape[1]-w_))

       
        i_2=int(i_1+mot_xy[i_1,j_1,0])
        j_2=int(j_1+mot_xy[i_1,j_1,1])
        q1=seg_map_1[i_1,j_1]
        q0=seg_map[i_2,j_2]
        if q1==q0:
            U_sketch1=B_1[i_1-w_0:i_1+w_0,j_1-w_0:j_1+w_0,:]
            U_sketch2=B_0[i_2-w_0:i_2+w_0,j_2-w_0:j_2+w_0,:]  
            if U_sketch2.shape ==(w_, w_, 1):
                response=np.minimum(np.sum(U_sketch2),np.sum(U_sketch1))-3
                U_sketch1=(seg_map_1[i_1-w_0:i_1+w_0,j_1-w_0:j_1+w_0]/l_+ B_1[i_1-w_0:i_1+w_0,j_1-w_0:j_1+w_0,0])/2-0.5

                U_sketch2=(seg_map[i_2-w_0:i_2+w_0,j_2-w_0:j_2+w_0]/l_+B_0[i_2-w_0:i_2+w_0,j_2-w_0:j_2+w_0,0] )/2-0.5 
                
                
                 
            else:
                response=-1
        else:
            U_sketch1=0
            U_sketch2=0
            response=-1
            
        
    except:
        U_sketch1=0
        U_sketch2=0
        response=-1
    return U_sketch1,U_sketch2,response




###################################################################


def bregman_approx_classik_01(factor_02_,compl_,measure_,kk_,hrd_tr):

    d_k=np.zeros((compl_.shape[0],1))
    YY1=1.0*factor_02_
    for tgd in range(1,kk_+1)  :
        L_S=np.dot(compl_,YY1.transpose())
        W_=np.dot(compl_,compl_.transpose())
        W_I=W_+ measure_* eye(W_.shape)
        W_inv=quasy_inv_matr_002(  W_I  ,0.00001)
        a1= measure_*d_k
        a2=np.mat(L_S)+np.mat(a1)
        coeff_09096=np.dot(W_inv,a2)
        d_k=np.mat(hrd_thr_oper(to_1D(coeff_09096), hrd_tr)).transpose()
    ###
    b1=compl_.transpose()
    b2=np.dot(b1,coeff_09096)
    approximation=b2.transpose()
    II_bregman_=mat.find(abs(d_k)>0); II_bregman_=to_1D(II_bregman_)
    best_gaussian_0=np.mat(compl_)[II_bregman_,:]
    
    return d_k,coeff_09096,approximation,best_gaussian_0

def bregman_approx_pozitive_00(factor_02_,compl_,measure_,kk_,hrd_tr):

    d_k=np.zeros((compl_.shape[0],1))
    YY1=1.0*factor_02_
    for tgd in range(1,kk_+1)  :
        L_S=np.dot(compl_,YY1.transpose())
        W_=np.dot(compl_,compl_.transpose())
        W_I=W_+ measure_* eye(W_.shape)
        W_inv=quasy_inv_matr_002(  W_I  ,0.00001)
        a1= measure_*d_k
        a2=np.mat(L_S)+np.mat(a1)
        coeff_09096=np.dot(W_inv,a2)
        d_k=np.mat(ReLU_01( hrd_thr_oper( to_1D(coeff_09096), hrd_tr),0)).transpose()
    ###
    b1=compl_.transpose()
    b2=np.dot(b1,coeff_09096)
    approximation=b2.transpose()
    II_bregman_=mat.find(abs(d_k)>0); II_bregman_= to_1D(II_bregman_)
    best_gaussian_0=np.mat(compl_)[II_bregman_,:]
    
    return d_k,coeff_09096,approximation,best_gaussian_0
def add_reflection_00(seq_gafol,w_):
    Q1=numpy.flip(seq_gafol[ -w_:,:,: ],0)
    P1=numpy.flip(seq_gafol[ :w_ ,:,: ],0)
    seq_gafol1=np.concatenate([P1,seq_gafol,Q1],0) 
    Q2=numpy.flip(seq_gafol1[ :,-w_:,:],1)
    P2=numpy.flip(seq_gafol1[ :,:w_,:],1)
    seq_gafol2=np.concatenate([P2,seq_gafol1,Q2],1) 
    return seq_gafol2

def add_reflection_01(seq_gafol,w_):
    Q1=numpy.flip(seq_gafol[:, -w_:,:,: ],1)
    P1=numpy.flip(seq_gafol[:, :w_ ,:,: ],1)
    seq_gafol1=np.concatenate([P1,seq_gafol,Q1],1) 
    Q2=numpy.flip(seq_gafol1[ :,:,-w_:,:],2)
    P2=numpy.flip(seq_gafol1[ :,:,:w_,:],2)
    seq_gafol2=np.concatenate([P2,seq_gafol1,Q2],2) 
    return seq_gafol2

def T_collect_00(seq_gafol, beg0,end0,beg1,end1, step_0,step_1,w_ ):

    Q1=numpy.flip(seq_gafol[:,-w_:,:,:],1)
    seq_gafol1=np.concatenate([seq_gafol,Q1],1) 
    Q2=numpy.flip(seq_gafol1[:,:,-w_:,:],2)
    seq_gafol2=np.concatenate([seq_gafol1,Q2],2) 
     

     
    collection_N=[]


    l_vector=w_**2 
    for iop in range( beg0,min(end0,seq_gafol.shape[1]) ,step_0):
        for ert in range(beg1,min(end1,seq_gafol.shape[2]) ,step_0):
            x_= iop 
            y_= ert 
            seq_=seq_gafol2[:,x_ :x_+w_,y_ :y_+w_,0]
            seq_vect=np.reshape(seq_,[seq_.shape[0],w_**2])
            collection_N.append(seq_vect)
    collection_N =np.array(collection_N) 
    T_= collection_N[::step_1,0,: ]  
    return T_

 

def  eig1dec(SYM):

    T_2,BASIS_ =np.linalg.eig(SYM )
    return BASIS_, np.diag(T_2)


def  eig0dec(SYM):

    T_2,BASIS_ =np.linalg.eig(SYM )
    return BASIS_, np.diag(T_2)


def norm(v):

    return np.sqrt(np.sum(np.square(v)))


def norm_strings(z):

    return np.array([norm(row) for row in z])


def norm_strings_2(z):

    return np.linalg.norm(z, axis=1)

def to_column(z):

    return  np.mat(to_1D(z)).transpose()

def to_str(z):

    return  np.mat(to_1D(z))

def points2ind_01(points_,size_):
    m=size_[0];
    II1= m*(np.mat(points_)[:,1])+np.mat(points_)[:,0]
    return    II1;
 
def quasy_inv_matr_000(collect_feature,qqq):

    U,S,V =np.linalg.svd(collect_feature )
    dd= S
    dd1=np.maximum(qqq,abs(dd))
    dd2=np.multiply(np.sign(dd),dd1)
    dd_=1.0/np.mat(dd2+0.0000001)
    S_= np.diag(to_1D(dd_ ))
    S1=S_
    n_=S_.shape[0]
    m_=S_.shape[1]
    S1[0:n_,0:m_]=1.0*S_
    collect_feature_inv=np.dot(np.dot(V.transpose(),S1),U.transpose())
    return collect_feature_inv,U,dd,V.transpose()


def quasy_inv_matr_002(collect_feature,qqq):

    return quasy_inv_matr_000(collect_feature,qqq)[0] 

def PrIntoSubsp_02(YY1,best_gaussian,esp_):
    L_S=np.dot(best_gaussian,YY1.transpose())
    W_=np.dot(best_gaussian,best_gaussian.transpose())
    W_inv=quasy_inv_matr_002(W_,esp_) 
    coeff_09096=np.dot(W_inv,L_S)
    approximation=(np.dot(best_gaussian.transpose(),coeff_09096)).transpose()

    return approximation,coeff_09096

def PrIntoSubsp_03(YY1,best_gaussian,esp_,eps1):
    L_S=np.dot(best_gaussian,YY1.transpose())
    W_=np.dot(best_gaussian,best_gaussian.transpose())+eps1 * np.eye(best_gaussian.shape[0]) 
    W_inv=quasy_inv_matr_002(W_,esp_) 
    coeff_09096=np.dot(W_inv,L_S)
    approximation=(np.dot(best_gaussian.transpose(),coeff_09096)).transpose()

    return approximation,coeff_09096

 
def  support_vector_00(galore,kkk,show):
    i0 = 0
    ii=[i0]      
    n_ = np.sqrt(np.sum(pow(galore, 2), 0))
    norm_ = np.array([1])
    for thf in range(1,kkk+1):
        Baz = np.array(galore[ii,:])
        [projection,coeff_09096]= PrIntoSubsp_02(galore,Baz,0.001) 
        ort_compl=galore-projection
        n_orth=np.array([np.sqrt(np.sum(pow(ort_compl,2),1))])
        n_orth=n_orth.transpose()
        ma = np.max(n_orth,0)
        ima = np.argmax(n_orth,0)
        ii=np.hstack((ii,ima))
        norm_=norm_.transpose()
        norm_=np.vstack((norm_, ma)) 
        norm_=norm_.transpose()

        if show:

            plt_multi_01_(n_orth.transpose(),'n_orth' )
            plt_multi_01_(norm_,'norm_')

    return Baz,ii,norm_
 
def sigmoid_01(x):
    return 1 / (1 +  cv2.exp(-x))

def ReLU(x):
    return  cv2.max(x, 0) 

def Conv3D_00(x, W):
    (wrow, wcol, numFilters) = W.shape
    (xrow, xcol, ) = x.shape
    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1
    y = np.zeros((yrow, ycol, numFilters))
    k=0
    for k in range(numFilters):
        fil = W[:, :, k]
        fil = np.rot90(np.squeeze(fil), 2)
        y[:, :, k] = scipy.signal.convolve2d(x, fil, mode='valid')
    return y

def Conv3D_02(x, W):
    (wrow, wcol, numFilters) = W.shape
    int(np.fix( (wcol-1)/2))
    anc=(int(np.fix( (wcol-1)/2)),int(np.fix( (wrow-1)/2)) )
    qq= x.shape
    xrow=qq[0]
    xcol=qq[1]
    
    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1
    y = np.zeros((xrow, xcol, numFilters))
   
    for k in range(numFilters):
        fil = W[:, :, k]
        #fil=np.array(fil).T 
        fil = np.rot90( fil , 2)
        
        filtered = cv2.filter2D( x, -1,  fil ,anchor =anc  )
        y[:, :, k] = filtered
    return y

def predict_feller_00(gray_frame,F_0456,feller_):
    start = time.time()
    y=Conv3D_02( gray_frame.astype('float') , F_0456)
    end = time.time()
    print("[INFO] Conv3D took {:.6f} seconds".format(end - start))
    print(F_0456.shape)
    V00=np.reshape(y,( -1,y.shape[2] ) ,order="F")
    start = time.time()

    for k in range(feller_['feller_k'][0].shape[0]-2):
        if feller_['feller_k'][0][k][ 'activation'][0][0][0]=='relu':
            #print('relu')
            #V01=np.maximum(V00, 0) 
            V01=ReLU_01(V00, 0)
        elif feller_['feller_k'][0][k][ 'activation'][0][0][0]=='sgmd':
            #print('sgmd')
            V01=sigmoid_01(V00 ) 
        Wk=  feller_['feller_k'][0][k+1][ 'W'][0][0]
        V11=np.dot(V01,Wk )
        #V11=cv2.multiply(V01,Wk )
        V00=V11
    k_end=feller_['feller_k'][0].shape[0]-2    
    if feller_['feller_k'][0][k_end][ 'activation'][0][0][0]=='relu':
            #print('relu')
            V01=ReLU_01(V00, 0)
    elif feller_['feller_k'][0][k_end][ 'activation'][0][0][0]=='sgmd':
        #print('sgmd')
        V01=sigmoid_01(V00 )     

    V11= np.dot((V01-0.5),feller_['C_final'])
    map_=cv2.max(np.reshape(V11,gray_frame.shape ,order="F")-0.5,0)

    end = time.time()    
    print("[INFO] rest took {:.6f} seconds".format(end - start))
    return map_

def predict_feller_01(G,F_0456,feller_):
    start = time.time()
    y=Conv3D_02(G.astype('float') , F_0456)
    end = time.time()
    print("[INFO] Conv3D took {:.6f} seconds".format(end - start))
    print(F_0456.shape)
    V00=np.reshape(y,( -1,y.shape[2] ) ,order="F")
    start = time.time()

    for k in range(feller_[0].shape[0]-2):
        print(feller_[0][k][ 'activation'][0][0][0])
        if feller_[0][k][ 'activation'][0][0][0]=='relu':
            #print('relu')
            #V01=np.maximum(V00, 0) 
            V01=ReLU_01(V00, 0)
        elif feller_[0][k][ 'activation'][0][0][0]=='sgmd':
            #print('sgmd')
            V01=sigmoid_01(V00 ) 
        Wk=  feller_[0][k+1][ 'W'][0][0]
        V11=np.dot(V01,Wk )
        #V11=cv2.multiply(V01,Wk )
        V00=V11
    k_end=feller_[0].shape[0]-2
    print(feller_[0][k_end][ 'activation'][0][0][0])
    if feller_[0][k_end][ 'activation'][0][0][0]=='relu':
            print('relu')
            V01=ReLU_01(V00, 0)
    elif feller_[0][k_end][ 'activation'][0][0][0]=='sgmd':
        #print('sgmd')
        V01=sigmoid_01(V00 )     

 
    map_= np.reshape(V01,G.shape ,order="F")

    end = time.time()    
    print("[INFO] rest took {:.6f} seconds".format(end - start))
    return map_

def Sigmoid(x):
    y=1.0 / (1 + np.exp(-x))
    return y

def RePool_01(e3, pool_const):
    (n1, n2, n3) = e3.shape
    e2 = np.zeros((n1 * pool_const, n2 * pool_const, n3))
    W3 = np.ones((n1 * pool_const, n2 * pool_const, n3)) / (pool_const**2)
    for c in range(n3):
        e2[:, :, c] = np.kron(e3[:, :, c],np.ones((pool_const, pool_const))) * W3[:, :, c]
    return e2

def Pool_01(x, q):
    (xrow, xcol, numFilters) = x.shape
    y = np.zeros((round(xrow / q),
                  round(xcol / q),
                  numFilters))
    for k in range(numFilters):
        fil = np.ones((q, q)) / (q**2)
        image = scipy.signal.convolve2d(x[:, :, k], fil, mode='valid')
        y[:, :, k] = image[::q, ::q]
    return y

def ReLU_01(x, q):
    x[x < q] = q
    return x

def Conv3D_00(x, W):
    (wrow, wcol, numFilters) = W.shape
    (xrow, xcol, ) = x.shape
    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1
    y = np.zeros((yrow, ycol, numFilters))
    k=0
    for k in range(numFilters):
        fil = W[:, :, k]
        fil = np.rot90(np.squeeze(fil), 2)
        y[:, :, k] = scipy.signal.convolve2d(x, fil, mode='valid')
    return y

def Conv3D_01(x, W):
    (wrow, wcol, numFilters) = W.shape
    (xrow, xcol, ) = x.shape
    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1
    y = np.zeros((yrow, ycol, numFilters))
    k=0
    for k in range(numFilters):
        fil = W[:, :, k]
         
        y[:, :, k] = scipy.signal.convolve2d(x, fil, mode='valid')
    return y



def Predict_MnistConv_Sgmd(x, W1, W5, Wo, param):
    relu_const_1 = param['relu_const_1']
    relu_const_2 = param['relu_const_2']
    pool_const = int(param['pool_const'])
    y1 = Conv3D_00(x, W1)
    y2 = ReLU_01(y1, relu_const_1)
    y3 = Pool_01(y2, pool_const)
    y4 = np.reshape(y3, (-1, 1), order='F')
    v5 = np.dot(W5, y4)
    y5 = ReLU_01(v5, relu_const_2)
    v = np.dot(Wo, y5)
    y = Sigmoid(v)
    return y

def Predict_MnistConv_Sgmd_01(x, W1, W5, Wo, param):
    relu_const_1 = param['relu_const_1']
    relu_const_2 = param['relu_const_2']
    pool_const = int(param['pool_const'])
    y1 = Conv3D_01(x, W1)
    y2 = ReLU_01(y1, relu_const_1)
    y3 = Pool_01(y2, pool_const)
    y4 = np.reshape(y3, (-1, 1), order='F')
    v5 = np.dot(W5, y4)
    y5 = ReLU_01(v5, relu_const_2)
    v = np.dot(Wo, y5)
    y = Sigmoid(v)
    return y


def UpdateW_MnistConv_Sgmd_01(W1, W5, Wo, Images, Labels,param,epoch):
    num = param['num']
    sparsity_ = param['sparsity_']
    alpha = param['alpha']
    beta = param['beta']
    mu_ = param['mu_']
    relu_const_1 = param['relu_const_1']
    relu_const_2 = param['relu_const_2']
    show=param['show']
    pool_const = 2
    idx_ = np.random.permutation(np.size(Images, axis=2))
    idx = np.array([idx_[0:min(num, np.size(Images, axis=2))]])
    X_ = Images[:, :, idx[0,:] -1]
    Y_ = Labels[:, idx[0,:]-1 ]
    VV1 = rand_ones_sparse_00(W1, sparsity_)
    VV5 = rand_ones_sparse_00(W5, sparsity_)
    VVo = rand_ones_sparse_00(Wo, sparsity_)
    d1_k = W1
    d5_k = W5
    do_k = Wo
    momentum1 = np.zeros((np.size(W1, axis=0),np.size(W1, axis=1),np.size(W1, axis=2)))
    momentum5 = np.zeros((np.size(W5, axis=0),np.size(W5, axis=1)))
    momentumo = np.zeros((np.size(Wo, axis=0),np.size(Wo, axis=1)))

    N = np.size(idx, axis=1)
    bsize = param['bsize']
    blist = np.array(range(1, N - bsize + 1, bsize))
    for batch in range(len(blist.T)):
        dW1 = np.zeros((np.size(W1, axis=0),np.size(W1, axis=1),np.size(W1, axis=2)))
        dW5 = np.zeros((np.size(W5, axis=0),np.size(W5, axis=1)))
        dWo = np.zeros((np.size(Wo, axis=0),np.size(Wo, axis=1)))
        begin = blist[batch]
        for k in range(begin,begin + bsize):
            x = X_[:,:, k-1]
            y1=Conv3D_00(x, W1)
            y2 = ReLU_01(y1, relu_const_1)
            y3 = Pool_01(y2, pool_const)
            y4 = np.reshape(y3.T, (-1,1))
            v5 = np.dot(W5 ,y4)
            y5 = ReLU_01(v5, 0)
            v  = np.dot(Wo,y5)
            prediction_ = Sigmoid(v)
            e = np.reshape(np.array(Y_[:,k-1]), (-1,1))
            e = e-prediction_
            delta = e
            e5 = np.dot(Wo.T,delta)
            delta5 = (y5 > relu_const_2) * e5
            e4 = np.dot(W5.T,delta5)
            e3 = np.reshape(e4, y3.shape, order="F")
            e2 = RePool_01(e3, pool_const)
            delta2 = (y2 > relu_const_1) * e2
            delta_1 = Conv3D_00(x, delta2)
            dW1 = dW1 + delta_1
            y4=y4.T
            dW5 = dW5 + np.dot(delta5,y4)
            y5=y5.T
            dWo = dWo + np.dot(delta,y5)
        dW1 /=  bsize
        dW5 /= bsize
        dWo /= bsize
        momentum1 = alpha * dW1 + beta * momentum1
        W1 = W1 + momentum1 - mu_ * (W1 - d1_k)
        momentum5 = alpha * dW5 + beta * momentum5
        W5 = W5 + momentum5 - mu_ * (W5 - d5_k)
        momentumo = alpha * dWo + beta * momentumo
        Wo = Wo + momentumo - mu_ * (Wo - do_k)
        d1_k = np.empty(W1.shape)
        for i in range(W1.shape[2]):
            d1_k[:, :, i] = W1[:, :, i] * VV1
        d5_k = VV5*W5
        do_k = VVo*Wo
        if show > 0:
            fig = plt.figure()
            fig.suptitle('epoch=%i' %epoch , fontsize=11)
            a = fig.add_subplot(2, 2, 1)
            imgplot1 = plt.imshow(W1[:, :, 1])
            plt.ylabel('batch=%i' % batch);
            a = fig.add_subplot(2, 2, 2)
            imgplot2 = plt.imshow(W1[:, :, 2])
            plt.ylabel('batch=%i' % batch);
            a = fig.add_subplot(2, 2, 3)
            imgplot3 = plt.imshow(W1[:, :, 3])
            plt.ylabel('batch=%i' % batch);
            a = fig.add_subplot(2, 2, 4)
            imgplot4 = plt.imshow(W1[:, :, 4])
            plt.ylabel('batch=%i' % batch);
            plt.show()
            UU = X_[:,:, begin: begin + bsize]
            Y_true = Y_[:, begin: begin + bsize]
            m = [Predict_MnistConv_Sgmd(UU[:,:, kjh], W1, W5, Wo, param) for kjh in range(0,np.size(UU, axis=2))]
            pp=np.array(m)
            pp=pp[:,:,0]
            fig = plt.figure()
            fig= plt.subplots(figsize=(10, 5))
            plt.clf();
            plt.plot(pp[:], 'r');
            plt.plot(Y_true[:].T, 'k');
            plt.title('epoch=%i' %epoch );
            plt.ylabel('k=%i' %k);
            plt.show()
            print('epoch=',epoch,'k=',k)
            wait = input("PRESS ENTER TO CONTINUE.")
    return W1, W5, Wo


def ai_2(img):
    mmin = np.min(img)
    mmax = np.max(img)
    plt.imshow(255-255*(img-mmin)/(mmax-mmin), cmap = 'Greys')
    plt.show()
    return
################################################################################33
def color_mask_descriptor_01(im ,model,k):
    # k-шаг гистограммы [0:k:255]
    y =predict_mask_model_00(im ,model)
    i ,mask =apply_mask_01( im ,y ,2)# 2-степень в которую мы возводим маску для контрастности
    color_image_hvs_  = cv2.cvtColor(im.astype('uint8'), cv2.COLOR_RGB2HSV) 
    # HSV цветовая модель. H=color_image_hvs_1[:,:,0]-оттенки цвета
    hs=np.histogram(color_image_hvs_[:,:,0], bins=k, range=None, normed=None, weights=mask, density=True) [0]
    return hs,mask,i


def mahal(y, x):
    (rx, cx) = x.shape
    (ry, cy) = y.shape

    if cx != cy:
        raise Exception('stats:mahal:InputSizeMismatch')
    if rx < cx:
        raise Exception('stats:mahal:TooFewRows')
    m = np.mean(x, 0)
    m_y_rows = np.repeat(m.reshape((1, cx)), ry, axis=0)
    m_x_rows = np.repeat(m.reshape((1, cx)), rx, axis=0)
    c = x - m_x_rows
    q, r = scipy.linalg.qr(c, mode='economic')
    ri = np.linalg.solve(r.T, (y - m_y_rows).T)
    return np.ravel(np.multiply(ri, ri).sum(axis=0).T * (rx - 1))
################ diff map  diffusion ###########################
def quasydiff2D_01(a,b):
    return a-b
def gaussian_kernel_function_dff(epsilon):
    def kernel_function(a, b):
        return np.exp(-(np.linalg.norm( quasydiff2D_01(a,b)) ** 2 / epsilon ** 2))
    def kernel_function_1(a, b):
        return np.linalg.norm( quasydiff2D_01(a,b))

    return kernel_function

def gaussian_kernel_function_dff_(epsilon):
    def kernel_function(a, b):
        return np.exp(-(np.linalg.norm( quasydiff2D_01(a,b),axis=-1) ** 2 / epsilon ** 2))
    def kernel_function_1(a, b):
        return np.linalg.norm( quasydiff2D_01(a,b))

    return kernel_function

def plot_matrix(matrix):
    return plt.imshow(matrix, cmap='binary')

def distance_vec_multi(new, base, distance_function):
    n = base.shape[0]
    dm = np.zeros([new.shape[0], n])
     
    
    for j in range(new.shape[0]):
        cur=np.expand_dims(new[j,:],0)
        diff= distance_function(np.tile(cur,[n,1]), base) 
        #print(np.tile(cur,[n,1]).shape,diff.shape)
        dm[j, :] = diff
    return dm
 
def distance_matrix(coords, distance_function):
    n = coords.shape[0]
    dm = np.zeros([n, n])
    for i, j in combinations(range(n), 2):
        dm[i, j] = distance_function(coords[i], coords[j])
        dm[j, i] = dm[i, j] # матрица симметричная
    for i in range(n):
        dm[i, i] = distance_function(coords[i], coords[i]) # диагональ может быть не нулевой

        
    return dm

def distance_vec(new, base, distance_function):
    n = base.shape[0]
    dm = np.zeros([1, n])
     
    for i in range(n):
        dm[0, i] = distance_function(new, base[i]) 
    return dm

def distance_vec_multi(new, base, distance_function):
    n = base.shape[0]
    dm = np.zeros([new.shape[0], n])
     
    
    for j in range(new.shape[0]):
        cur=np.expand_dims(new[j,:],0)
        diff= distance_function(np.tile(cur,[n,1]), base) 
        #print(np.tile(cur,[n,1]).shape,diff.shape)
        dm[j, :] = diff
    return dm

def create_DM_00(data_,eps_,k_):
    kernel_function = gaussian_kernel_function_dff_(eps_) 
    #similarity_matrix = distance_matrix(data_, kernel_function)
    similarity_matrix =distance_vec_multi(data_, data_, kernel_function)
    L_=similarity_matrix
    eig_values , eig_vectors =diffusion_map_00(L_,k=k_)
    eig_vectors=np.real(eig_vectors)

    DM={'data_':data_,'kernel_function':kernel_function,'eig_vectors':eig_vectors,'eig_values':eig_values}
    return DM

def insert_into_DM_00(vec_new,DM):
    cogn_vect=distance_vec_multi(vec_new, DM['data_'], DM['kernel_function'])
    project_DM = np.real(np.dot( cogn_vect,DM['eig_vectors']))/DM['eig_values'] 
    return project_DM



def labels2partition_00(data_,labels_,DM,k_):
    subclasses=[]
    for l_ in set(labels_ ):
        ii_l=np.where(labels_==l_)[0]
        if len(ii_l)>k_:
            data_l=np.real(data_[ii_l,:])
            project_DM1=insert_into_DM_00(data_l,DM)
            dd=dispers_str_01(data_l) 
            #plot_im(m_)
            class_={'data_':data_l, 'l_':l_, 'DM':project_DM1,\
                    'i_reg':None, 'i_anom':None, \
                    'centrum':dd[0],'dispers':dd[1],'label': str(100000+l_)[-3:] }
            subclasses.append(class_)
    return subclasses
def subclasses2simple(subclasses,key):
    subcl=[]
    for l_ in subclasses:
        da_=l_[key]
        #da_=l_['DM']
        subcl.append(da_)
    return subcl


def plot_3d_2(coords,new):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2] ,marker='*',  c='r',label = 0)
    ax.scatter(new[:, 0], new[:, 1], new[:, 2] ,marker='o',c='g',label = 1)

def plot_3d_3(coords,new1,new2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],c='r',label = 0 )
    ax.scatter(new1[:, 0], new1[:, 1], new1[:, 2] ,c='k',label = 1)
    ax.scatter(new2[:, 0], new2[:, 1], new2[:, 2],c='g',label = 2 )

def diffusion_map_00(markov_chain, k=3):
    eig_values, eig_vectors = np.linalg.eig(markov_chain)
    k_most = np.argsort(np.abs(eig_values))[-k:]
    return eig_values[k_most], eig_vectors[:, k_most]#* eig_values[k_most]

def plot_3d(coords):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])




def diffusion_representation_helek3(GG,n1,n2,str,sigma):

    similarity_=np.zeros((GG.shape[1-1],GG.shape[1-1]))
    for iii in range(1,GG.shape[1-1]+1)  :
        AA=np.mat(GG)[iii-1,:]
        for jjj in range(1,GG.shape[1-1]+1)  :
            BB=np.mat(GG)[jjj-1,:]
            if(str=='l2')  :
                similarity_[iii-1,jjj-1]=np.linalg.norm(AA-BB)
            elif(str=='sc')  :
                similarity_[iii-1,jjj-1]=np.dot(AA,BB.transpose())
            elif(str=='l1')  :
                similarity_[iii-1,jjj-1]=(AA-BB).max()
            elif(str=='df')  :
                similarity_[iii-1,jjj-1]=1-np.exp(-np.linalg.norm(AA-BB)/sigma)
            ###
        ###
    ###
    BASIS_,T_a=eig0dec(similarity_)
    T_=abs(np.diag(T_a))
    T_1,II=sort1(T_,-1)
    SS=-np.mat(BASIS_)[:,II[n1:n2]].transpose()
    T=T_[II[n1-1:n2]-1]
    return SS,T,similarity_
################# analit geometry #################################3
 
view=[-12,45]

def random_choose_2_00(data1,data2,kama_represento):
    ll_= data1.shape[0]
    rr_ind_0=np.array([np.random.permutation(ll_)])[0] 
    rr_ind_1=rr_ind_0[:kama_represento]
    data_random_ind=data1[rr_ind_1,:] 
    data_norm_ind=data2[rr_ind_1,:] 
    return data_random_ind,data_norm_ind
def conzentration(X_0, Pnt):
    #log("Start find conzentration for {}".format(X_0))
    X_k = X_0
    SW_k = 1
 
    eps_=  0.35
    for oi in range(10):
        for io7 in range(len(Pnt)):
            cl_i = Pnt[io7,:]
            W_k = np.exp(-np.linalg.norm(X_k - cl_i)**2 / eps_**2)
             
            #print("W_k={} in={} norm={} norm**2={}".format(W_k, -np.linalg.norm(X_0 - cl_i)**2 / eps_**2, np.linalg.norm(X_0 - cl_i), np.linalg.norm(X_0 - cl_i)**2))
            SW_k_p_1 = SW_k + W_k
            X_k_p_1 = X_k * (SW_k / (0.000000000001 + SW_k_p_1)) + (cl_i  * W_k) / SW_k_p_1
            X_k = X_k_p_1/np.linalg.norm(X_k_p_1)
            SW_k = SW_k_p_1
            #print("W_k={} SW_k={} X_k={} X_0={}".format(W_k, SW_k, X_k, X_0))
        #if oi % 10 == 0:
            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            #ax.scatter3D(Pnt[:, 0], Pnt[:, 1], Pnt[:, 2], c='black');
            #ax.scatter3D(X_k[0], X_k[1], X_k[2], c='red');
            #plt.show()
    #log("conzentration for {} is {}".format(X_0, X_k))
    return X_k

def represent_sphere_1(data_0,data_1,step,view):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
   
    ax.scatter3D(data_1[: , 0],data_1[: , 1], data_1[: , 2], '*k')
    ax.scatter3D(0, 0, 0, 'red')
    ax.scatter3D(data_0[::step, 0], data_0[::step, 1], data_0[::step, 2], '.black')

    ax.azim = view[0]
    ax.elev = view[1]
    plt.show()


def random_sphere_point(kama):
    x=[]
    for vbn in range(kama):

        n_0=np.random.normal(size=3)
        X_0=n_0/np.linalg.norm(n_0)
        x.append(X_0)
    return  np.array(x)

    

def represent_sphere_0(data_norm_ind,view):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data_norm_ind[:, 0], data_norm_ind[:, 1], data_norm_ind[:, 2], 'black')
    ax.scatter3D(0, 0, 0, 'red')

    ax.azim = view[0]
    ax.elev = view[1]
    plt.show()

def points_of_plane_00(data_3d_,N_,D_,thr_):
    project_=data_3d_@ N_  
    #plot_im(np.exp(-(project_-D_) **2/0.03**2))
    ind_ = ((project_-D_) **2< thr_).nonzero()[0]
    Plane_point=data_3d_[ind_ ,:]
    return Plane_point,ind_

def random_choose_1_00(data1, kama_represento):
    ll_= data1.shape[0]
    rr_ind_0=np.array([np.random.permutation(ll_)])[0] 
    rr_ind_1=rr_ind_0[:kama_represento]
    data_random_ind=data1[rr_ind_1,:] 
     
    return data_random_ind 


def plot_im(img):
    Y_vect=img.ravel()
    plt.figure(figsize=(25,10))

    plt.plot(Y_vect ,'k')
     
    plt.show() 
def plot_im_2(img,img2):
     
    plt.figure(figsize=(5,5))
    #plt.axes().set_aspect('equal')
    plt.plot(img.ravel() ,'k')
    plt.plot(img2.ravel() ,'r')
     
    plt.show()  
    
def plot_im_2a(img,img2):
     
    plt.figure(figsize=(5,5))
    plt.axes().set_aspect('equal')
    plt.plot(img.ravel() ,'k')
    plt.plot(img2.ravel() ,'r')
     
    plt.show()  

def random_sphere_point(kama):
    x=[]
    for vbn in range(kama):

        n_0=np.random.normal(size=3)
        X_0=n_0/np.linalg.norm(n_0)
        x.append(X_0)
    return  np.array(x)

def choose_concentrations_00(X_1,data_norm_ind,data_random_ind,show) :
    s_=set({})     
    plates=[]
    thr_=0.07
    for tyui in range(X_1.shape[0]):
        point_cnzntrz_0=X_1[tyui,:]
        Q1 = data_norm_ind - point_cnzntrz_0
        Q2 = Q1 ** 2
        Q3 = Q2.sum(axis=1)
        
        ind_ = (Q3 < thr_).nonzero()[0]
        intersect_= s_ & set(ind_) 
        l_=len(set(ind_))
        q_= len(intersect_) / l_
        #print(l_,q_)
        if (q_<0.5) and l_>200:  
            N_=np.mean(data_norm_ind[ind_,:],0)## нормаль
            T_=np.mean(data_random_ind[ind_,:],0)## точка привязки
            plates.append([T_,N_,ind_])
            s_= s_|set(ind_) 
            if show:
                ind_1 = (Q3>=thr_).nonzero()[0]
                represent_sphere_1(data_norm_ind[ind_1,:],data_norm_ind[ind_ ,:],  5 ,view)
                represent_sphere_1(data_random_ind[ind_1,:],data_random_ind[ind_ ,:],  5 ,view)
                print('----------------------------------------')
    return plates
def choose_concentrations_01(X_1,data_norm_ind,thr_, kama_,show) :
    s_=set({})     
    plates=[]
     
    for tyui in range(X_1.shape[0]):
        point_cnzntrz_0=X_1[tyui,:]
        Q1 = data_norm_ind - point_cnzntrz_0
        Q2 = Q1 ** 2
        Q3 = Q2.sum(axis=1)
        #plot_im(Q3)
        ind_ = (Q3 < thr_).nonzero()[0]
        intersect_= s_ & set(ind_) 
        l_=len(set(ind_))+0.000000001
        q_= len(intersect_) / l_
        #print(l_,q_)
        if (q_<0.7) and l_>kama_:  
            X_0=np.mean(data_norm_ind[ind_,:],0) 
            
            plates.append([X_0,ind_])
            s_= s_|set(ind_) 
            if show:
                ind_1 = (Q3>=thr_).nonzero()[0]
                represent_sphere_1(data_norm_ind[ind_1,:],data_norm_ind[ind_ ,:],  5 ,view)
                 
                print('----------------------------------------')
    return plates


def conzentration_00(X_0, Pnt):
    #log("Start find conzentration for {}".format(X_0))
    X_k = X_0
    SW_k = np.ones((X_0.shape[0],1 ))
 
    eps_=  0.35
    for oi in range(10):
        for io7 in range(len(Pnt)):
            cl_i = Pnt[io7,:]
            W_k = np.expand_dims(np.exp(-np.linalg.norm((X_k - cl_i),axis=-1)**2 / eps_**2),1)
            #print(W_k.shape) 
            #print("W_k={} in={} norm={} norm**2={}".format(W_k, -np.linalg.norm(X_0 - cl_i)**2 / eps_**2, np.linalg.norm(X_0 - cl_i), np.linalg.norm(X_0 - cl_i)**2))
            SW_k_p_1 = SW_k + W_k
            #print(X_k.shape)
            #print(SW_k.shape)
            
            
            X_k_p_1 = X_k * np.divide(SW_k ,(0.000000000001 + SW_k_p_1)) + (cl_i  * W_k) / SW_k_p_1
            X_k = X_k_p_1/np.expand_dims(np.linalg.norm(X_k_p_1,axis=-1),1)
            SW_k = SW_k_p_1
            #print("W_k={} SW_k={} X_k={} X_0={}".format(W_k, SW_k, X_k, X_0))
        #if oi % 10 == 0:
            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            #ax.scatter3D(Pnt[:, 0], Pnt[:, 1], Pnt[:, 2], c='black');
            #ax.scatter3D(X_k[0], X_k[1], X_k[2], c='red');
            #plt.show()
    #log("conzentration for {} is {}".format(X_0, X_k))
    return X_k

def conzentration_01(X_0, Pnt,eps_):
    #log("Start find conzentration for {}".format(X_0))
    X_k = X_0
    SW_k = np.ones((X_0.shape[0],1 ))
 
     
    for oi in range(10):
        for io7 in range(len(Pnt)):
            cl_i = Pnt[io7,:]
            W_k = np.expand_dims(np.exp(-np.linalg.norm((X_k - cl_i),axis=-1)**2 / eps_**2),1)
            #print(W_k.shape) 
            #print("W_k={} in={} norm={} norm**2={}".format(W_k, -np.linalg.norm(X_0 - cl_i)**2 / eps_**2, np.linalg.norm(X_0 - cl_i), np.linalg.norm(X_0 - cl_i)**2))
            SW_k_p_1 = SW_k + W_k
            #print(X_k.shape)
            #print(SW_k.shape)
            
            
            X_k_p_1 = X_k * np.divide(SW_k ,(0.000000000001 + SW_k_p_1)) + (cl_i  * W_k) / SW_k_p_1
            X_k = X_k_p_1/np.expand_dims(np.linalg.norm(X_k_p_1,axis=-1),1)
            SW_k = SW_k_p_1
            #print("W_k={} SW_k={} X_k={} X_0={}".format(W_k, SW_k, X_k, X_0))
        #if oi % 10 == 0:
            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            #ax.scatter3D(Pnt[:, 0], Pnt[:, 1], Pnt[:, 2], c='black');
            #ax.scatter3D(X_k[0], X_k[1], X_k[2], c='red');
            #plt.show()
    #log("conzentration for {} is {}".format(X_0, X_k))
    return X_k

 
def correct_plane_00(T0,N0,data_3d_):
    T_k=T0
    N_k=N0


    LambdaT = 0.3
    LambdaV = 100.3
    Mu = 0.8
    sigma_ = 0.01
    DN_k = N_k
    for lih in range(100):
        V_k = data_3d_ - np.tile(T_k, (len(data_3d_), 1))
        Err_Vect=(V_k @ N_k) / np.linalg.norm(N_k)
        W_ = np.exp(- Err_Vect**2 / sigma_**2)
        
        
        V1_k = V_k * np.tile(W_, (3, 1)).T
        l_1 = np.sum(W_)
        DeltaV = (V1_k.T @ V1_k / l_1**2) @ N_k
        DeltaT = np.mean(Err_Vect * W_)

        T_k = T_k + LambdaT * (N_k / np.linalg.norm(N_k)) * DeltaT
        N_k = N_k - LambdaV * DeltaV + Mu * (DN_k - N_k)
        DN_k = N_k / np.linalg.norm(N_k);

        stop_t=DeltaT 
        stop_v=np.linalg.norm(DeltaV)/np.linalg.norm(N_k) 

        if max(stop_t,stop_v) < 0.0000001:
            break

    N_ = N_k / np.linalg.norm(N_k) 
    D_ = T_k @ N_
    print(plates[rty][1],N_)
    # plane equation N_*X=D_
    return N_,D_

def correct_plane_01(T0,N0,data_3d_,sigma_ = 0.1):
    T_k=T0
    N_k=N0


    LambdaT = 0.3
    LambdaV = 2.3
    Mu = 0.8
    
    DN_k = N_k
    for lih in range(100):
        V_k = data_3d_ - np.tile(T_k, (len(data_3d_), 1))
        Err_Vect=(V_k @ N_k) / np.linalg.norm(N_k)
        W_ = np.exp(- Err_Vect**2 / sigma_**2)
        
        
        V1_k = V_k * np.tile(W_, (3, 1)).T
        l_1 = np.sum(W_)
        DeltaV = (V1_k.T @ V1_k / l_1**2) @ N_k
        DeltaT = np.mean(Err_Vect * W_)

        T_k = T_k + LambdaT * (N_k / np.linalg.norm(N_k)) * DeltaT
        N_k = N_k - LambdaV * DeltaV + Mu * (DN_k - N_k)
        DN_k = N_k / np.linalg.norm(N_k);

        stop_t=DeltaT 
        stop_v=np.linalg.norm(DeltaV)/np.linalg.norm(N_k) 

        if max(stop_t,stop_v) < 0.0000001:
            break

    N_ = N_k / np.linalg.norm(N_k) 
    D_ = T_k @ N_
     
    # plane equation N_*X=D_
    return N_,D_

#######################################################################
def complete_files_01(dir_name,m,n):
    folder = []
    q=os.listdir(dir_name)
    l_=len(q)
    q1=np.array(random.sample(q,min(l_,m)))
    
    for folder_ in q1:
        folder.append(dir_name+folder_)
        #print(folder_)
    numbers = []
    number = 0
    files = []
    for fold in folder:
        q=os.listdir(fold + '/')
         
        q1=np.array(random.sample(q,min(len(q),n)))
        for file_ in q1:
            files.append(fold+ '/' +file_)
            numbers.append(number)
        number += 1
     
    
    return folder, files, numbers

def vizu_file(file_):
         
        stream = open(file_, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()

        bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 
         
        if( bw_.shape[0]==256 ) and (bw_.shape[1]==256) :
             
            resize_image = downscale_2(bw_)
        else:
            resize_image = cv2.resize(bw_, (128,128), interpolation = cv2.INTER_AREA)
        fig = plt.figure(figsize=(5, 5)) 
        plt.axes().set_aspect('equal')
        ai_2(resize_image)

def downscale_2( img):
    img = img.astype(np.uint16)
    img = img[:, 0::2] + img[:, 1::2]
    img = img[0::2, :] + img[1::2, :]
    img >>= 2
    return img.astype(np.float)    

def vectorize_fls(files,TL_001):
    features=[]
    len_files=min(10000000,len(files))
    for rty in range(len_files):
        file_=files[rty]
        stream = open(file_, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()

        bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 
         
        if( bw_.shape[0]==256 ) and (bw_.shape[1]==256) :
             
            resize_image = downscale_2(bw_)
        else:
            resize_image = cv2.resize(bw_, (128,128), interpolation = cv2.INTER_AREA)

        image_dims = np.expand_dims(resize_image/255, axis=[0,3])
        x,_=TL_001.forward_eshar_00(image_dims , image_dims)
        #plot_im(x)
        features.append(x[0,:])
    features  = np.array( features)
    return features
    
def complete_files_02(dir_name,m,n):
    folder = []
    q=os.listdir(dir_name)
    l_=len(q)
    print('os.listdir(dir_name)',l_)
    q1=np.array(random.sample(q,min(l_,m)))
    
    for folder_ in q1:
        folder.append(dir_name+folder_)
        #print(folder_)
    numbers = []
    number = 0
    files = []
    for fold in folder:
        q=os.listdir(fold + '/')
         
        q1=np.array(random.sample(q,min(len(q),n)))
        for file_ in q1:
            files.append(fold+ '/' +file_)
            numbers.append(number)
        number += 1
     
    
    return folder, files, numbers


def vectorize_fls_crop(files,TL_001):
    features=[]
    len_files=min(10000000,len(files))
    for rty in range(len_files):
        file_=files[rty]
        stream = open(file_, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()

        bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 
        if( bw_.shape[0]==256 ) and (bw_.shape[1]==256) :
            #cropped_im=bw_[  192-100: 192+100,128-100:128+100]
            cropped_im=bw_[ 192-42: 192+24,128-80:128+80]
            try:
                resize_image = cv2.resize(cropped_im, (128,128), interpolation = cv2.INTER_AREA)
            except:
                print(pimage)
                print(cropped_im.shape)
        if( bw_.shape[0]==128 ) and (bw_.shape[1]==128) :
            cropped_im=bw_[ 96-21: 96+12,64-40:64+40]
            try:
                resize_image = cv2.resize(cropped_im, (128,128), interpolation = cv2.INTER_AREA)
            except:
                print(pimage)
                print(cropped_im.shape)
        image_dims = np.expand_dims(resize_image, [0,3])/250
         
        
        x,_=TL_001.forward_eshar_00(image_dims , image_dims)
        #plot_im(x)
        features.append(x[0,:])
    features  = np.array( features)
    return features
def vectorize_fls_crop_01(fold,files,TL_001):
    features=[]
    len_files=min(10000000,len(files))
    for rty in range(len_files):
        file_=files[rty]
        stream = open(fold+'/'+file_, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()

        bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 
        if( bw_.shape[0]==256 ) and (bw_.shape[1]==256) :
            #cropped_im=bw_[  192-100: 192+100,128-100:128+100]
            cropped_im=bw_[ 192-42: 192+24,128-80:128+80]
            try:
                resize_image = cv2.resize(cropped_im, (128,128), interpolation = cv2.INTER_AREA)
            except:
                print(pimage)
                print(cropped_im.shape)
        if( bw_.shape[0]==128 ) and (bw_.shape[1]==128) :
            cropped_im=bw_[ 96-21: 96+12,64-40:64+40]
            try:
                resize_image = cv2.resize(cropped_im, (128,128), interpolation = cv2.INTER_AREA)
            except:
                print(pimage)
                print(cropped_im.shape)
        image_dims = np.expand_dims(resize_image, [0,3])/250
         
        
        x,_=TL_001.forward_eshar_00(image_dims , image_dims)
        #plot_im(x)
        features.append(x[0,:])
    features  = np.array( features)
    return features
#####################################
def dispers_str(a):
    return np.mean(a,0),np.mean(np.std(a,0))
def dispers_str_01(a):
    mean_=np.mean(a,0)
    diff=a-np.tile(mean_, (a.shape[0], 1))
    return mean_,np.mean(np.sqrt(np.mean(np.power(diff,2),1)))

##################################3
def entropy(labels):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)


def blur( imgarr):
    h, w, _ = imgarr.shape
    max_dim = h if h >= w else w
    min_ksize = 1 + round(max_dim / 128) if max_dim != 128 else 3  # для 128х128 3 - 5
    max_ksize = 9 + round(max_dim / 128) if max_dim != 128 else 5
    rand_ksize = random.randint(min_ksize, max_ksize)
    blur_ksize = round(rand_ksize // 3) if rand_ksize != 2 else 1  # для 128х128 1 - 2
    imgarr = cv2.blur(imgarr, (blur_ksize, blur_ksize), cv2.BORDER_DEFAULT)
    #print(blur_ksize, blur_ksize)
    return imgarr
def noisy(  img_arr: np.ndarray) -> np.ndarray:
    # random_noise() method will convert image in [0, 255] to [0, 1.0],
    # inherently it use np.random.normal() to create normal distribution
    # and adds the generated noised back to image
    img_arr = random_noise(img_arr.copy(), mode="gaussian", var=0.005*np.random.rand())
    return (255 * img_arr).astype(np.uint8)
def rotate_img(  imgarr, angle):
    interpolation=cv2.INTER_CUBIC
    image_center = tuple(np.array(imgarr.shape[:2]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(imgarr, rot_mat, imgarr.shape[1::-1], flags=interpolation,
                            borderMode=cv2.BORDER_REFLECT_101, borderValue=(0, 0, 0,))
    return result
def flip_img( imgarr, flip):
    imgarr = cv2.flip(imgarr, flip)
    return imgarr
def adjust_gamma(  imgarr, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(imgarr, table)
def kernel_motion_blur(  kernel_size, angle):
    '''
    param filter_size: integer, size of square blur filter.
    param angle: integer from range [-90,90]. clockwise angle between horizontal line and target line
        angle=0 means vertical blur
        angle=90 means horizontal blur
        angle>0 means downleft - upright blur
        angle<0 means upleft - downright blur
    returns: filter array of 0s and 1s of size (filter_size x filter_size) with normalization koefficient 1/filter_size
    '''
    if angle > 90:
        angle = 90
    elif angle < -90:
        angle = -90
    kernel_size = int(kernel_size)
    ab_angle = abs(angle)
    start_point = (0, 0)
    end_point = (kernel_size, np.int(kernel_size * np.tan(np.radians(min(ab_angle, 90 - ab_angle)))))
    kernel_motion = np.zeros((kernel_size, kernel_size))
    kernel_motion = cv2.line(kernel_motion, start_point, end_point, 1, 1)
    kernel_motion = kernel_motion / kernel_size
    if angle < -45:
        return np.flip(np.transpose(kernel_motion), 0)
    elif angle >= -45 and angle < 0:
        return np.flip(kernel_motion, 1)
    elif angle >= 0 and angle <= 45:
        return kernel_motion
    else:
        return np.transpose(kernel_motion)

def motion_blur(  imgarr):
    h, w, _ = imgarr.shape
    max_dim = h if h >= w else w
    min_ksize = 3 + round(max_dim / 128) if max_dim != 128 else 3  # для 128х128 3 - 5
    max_ksize = 5 + round(max_dim / 128) if max_dim != 128 else 5
    rand_ksize = random.randint(min_ksize, max_ksize)
    rand_angle = random.randint(-90, 90)
    kernel_mb = kernel_motion_blur(rand_ksize, rand_angle)
    imgarr = cv2.filter2D(imgarr, -1, kernel_mb)

    blur_ksize = round(rand_ksize // 3) if rand_ksize != 2 else 1  # для 128х128 1 - 2
    imgarr = cv2.blur(imgarr, (blur_ksize, blur_ksize), cv2.BORDER_DEFAULT)
    return imgarr
def clipper_img ( img_arr: np.ndarray) -> np.ndarray:
    """
    Вырезать прямоугольную вертикальную область из изображения
    :param img_arr: np.ndarray
    :return: np.ndarray
    """
    (h, w) = img_arr.shape[:2]
    if h > w:
        nip_off_pecent = int(h *  np.random.rand()*0.5)
        cut_img = img_arr[nip_off_pecent:h - nip_off_pecent, :]
        (h_c, w_c) = cut_img.shape[:2]
        img_arr = cut_img[:h // 2, w // 2 - w // 4:w_c // 2 + w // 4]
    elif h < w:
        nip_off_pecent = int(w *  np.random.rand()*0.5)
        cut_img = img_arr[:, nip_off_pecent:w - nip_off_pecent]
        (h_c, w_c) = cut_img.shape[:2]
        img_arr = cut_img[:, w_c // 2 - ((h ** 2) // w) // 2:w_c // 2 + ((h ** 2) // w) // 2]
    else:
        nip_off_pecent = int(w * 0.12)
        cut_img = img_arr[nip_off_pecent:h - nip_off_pecent, :]
        img_arr = cut_img[:, w // 2 - ((h ** 2) // w) // 4:w // 2 + ((h ** 2) // w) // 4]
    return img_arr
def mix_imgs(  img_arr_1: np.ndarray,img_arr_2: np.ndarray, class_label: Union[int, float], alpha: float = None) -> np.ndarray:
    alpha = random.uniform(a=.3, b=.99) if alpha is None else alpha

     
    return cv2.addWeighted(src1=img_arr_1,
                           alpha=alpha, beta=1 - alpha, gamma=0,
                           src2=img_arr_2)
def warp_and_mix_imgs( img_arr: np.ndarray) -> np.ndarray:
        height, width = img_arr.shape[:2]
        src_ul = (0, 0)
        src_dl = (0, img_arr.shape[0])
        src_ur = (img_arr.shape[0], 0)
        src_dr = (img_arr.shape[0], img_arr.shape[0])
 
        random_counts_1 = np.random.randint(low=width * 0.001, high=width // 2 - width * 0.4, size=2)
        random_counts_2 = np.random.randint(low=height * 0.001, high=height // 2 - height * 0.4, size=2)
        random_counts_3 = np.random.randint(low=width // 2 + width * 0.4, high=width * 0.99, size=2)
        random_counts_4 = np.random.randint(low=height // 2 + height * 0.4, high=height * 0.99, size=2)
 
        dst_ul = (random_counts_1[0], random_counts_2[0])
        dst_dl = (random_counts_1[1], random_counts_4[0])
 
        dst_ur = (random_counts_3[0], random_counts_2[1])
        dst_dr = (random_counts_3[1], random_counts_4[1])
 
        pts_src = np.float32([src_ul, src_dl, src_ur, src_dr])
        pts_dst = np.float32([dst_ul, dst_dl, dst_ur, dst_dr])
 
        transform_mtrx = cv2.getPerspectiveTransform(pts_src, pts_dst)
 
        dst_img_arr = cv2.warpPerspective(src=img_arr, M=transform_mtrx, dsize=(height, width))
        mix_img_arr = mix_imgs(img_arr_1=dst_img_arr, img_arr_2=dst_img_arr,class_label=0.1, alpha=0.5)
        
        return mix_img_arr
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№

def conv1d_00(x,n):
    win = scipy.signal.windows.hann(n)
    y = scipy.signal.convolve(x, win, mode='same') / sum(win)
    return y
def local_filter(x, order):
    x.sort()
    return x[order]
def ordfilt1d(A, order, mask_size):
    #A.shape=(1, N)
    return nd_filters.generic_filter(A, lambda x, ord=order: local_filter(x, ord), size=[1,mask_size  ])
def local_max(x,n):
    x=np.expand_dims(x.ravel(),0)
    y=ordfilt1d(x, -1,n)
    i=np.where(y[0,:]==x[0,:])[0]
    return i
def local_max_01(x,n,q):
    x=np.expand_dims(x.ravel(),0)
    y=ordfilt1d(x, -1,n)
     
    #plot_im_2(x[0,:],y[0,:])
    i0=np.where( y[0,:]==x[0,:] )[0]
    
    i1=np.where( x[0,:]>q )[0]
     
    i2=list(set(i0) & set(i1))
     
    return i2

def amax_2D(X_scaled):
    i_sort_relation_ = np.argmax(X_scaled.ravel(), axis=-1)
    i_amax = int(np.fix(i_sort_relation_ / X_scaled.shape[1]))
    j_amax = int(i_sort_relation_ - i_amax * X_scaled.shape[1])
    return i_amax, j_amax, X_scaled[i_amax, j_amax]

def one_to_one_relation_02_vengersky(relation_):
    X_scaled = relation_.copy()  
    # print(np.round(100*X_scaled))
    correspond_ = []
    while (1):

        i_, j_, ma_ = amax_2D(X_scaled)
        # print("i_,j_,ma_",i_,j_,ma_)
        if ma_ < 0:
            break
        correspond_.append([i_, j_])
        X_scaled[i_, :] = -0.0001
        X_scaled[:, j_] = -0.0001

    correspond_0 = np.array(correspond_)
    return correspond_0


def relation_of_matr_00(CORR):
    corr=one_to_one_relation_02_vengersky(CORR)
    norm=[]
    for i_ in corr:
        norm.append(CORR[i_[0],i_[1]])
    return norm


################################################################################################3
def IOU_mask_00(a,b):

    result1=(a>0.5)
    result2=(b>0.5)
    sum_1=np.sum(result1)
    sum_2=np.sum(result2)
     
    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    iou_score = np.sum(intersection) / (np.sum(union)+0.000001)
    return iou_score,sum_1,sum_2
def IOU_mask_01(a_predict,b_true, thr):

    result1=(a_predict>0.5)
    result2=(b_true>0.5)
    sum_1=np.sum(result1)
    sum_2=np.sum(result2)
    if (sum_1<thr) and (sum_2<thr):
        iou_score=1
        false_positive=0
        not_recgnized=0
    elif (sum_1>=thr) and (sum_2<thr): 
        iou_score=0
        false_positive=1
        not_recgnized=0
    elif (sum_1<thr) and (sum_2>=thr): 
        iou_score=0
        false_positive=0
        not_recgnized=1
    elif (sum_1>=thr) and (sum_2>=thr): 
        intersection = np.logical_and(result1, result2)
        union = np.logical_or(result1, result2)
        iou_score = np.sum(intersection) / (np.sum(union)+0.000001)
        false_positive=0
        not_recgnized=0

    return iou_score,false_positive,not_recgnized

def conf_matr_of_segmentation_00_7(batch_A,batch_P,thr_presense,thr_IOU):
    conf_matr=np.zeros((7,7))

    for ghj in range(6):
        label_cur=batch_A[:,:,ghj]
        if np.sum(label_cur.ravel())>thr_presense: # есть воздействие
            #ai_2(label_cur)
            response_lab=[]
            for vbn in range(6):
                pred_cur=batch_P[:,:,vbn]
                if np.sum(pred_cur.ravel())>thr_presense: # есть отклик
                    response_lab.append(IOU_mask_00(label_cur,pred_cur)[0])                
                else:
                    response_lab.append(0) 
            i_=np.argmax(response_lab)

            maxnorm=response_lab[i_]
            if maxnorm>thr_IOU:
                conf_matr[i_,ghj]=1
            else:
                conf_matr[-1,ghj]=1

            #print(maxnorm)
            #print(response_lab)                            

        elif 0:  # нет воздействиz
            response_lab=[]

            pred_cur=batch_P[:,:,ghj]
            if np.sum(pred_cur.ravel())>thr_presense: # есть отклик
                conf_matr[ghj,-1]=1        

    #label_backr=batch_A[:,:,-1] 
    #ai_2(label_backr)
    label_backr=batch_A[:,:,-1]      
    for vbn in range(6):#проверяем ложные детекции
        pred_cur=batch_P[:,:,vbn]
        intersect=np.minimum(pred_cur,label_backr)
        #ai_2(intersect)
        #print(np.sum(intersect.ravel())/(np.sum(pred_cur.ravel())+0.00001))
        if np.sum(intersect.ravel())/(np.sum(pred_cur.ravel())+0.00001)>0.2: # есть отклик
            conf_matr[vbn,-1 ]=1                      





    if IOU_mask_00(batch_A[:,:,-1],batch_P[:,:,-1])[0]> 0.95:
        conf_matr[-1,-1]=1
    return conf_matr

##############3  Polynom approx  ########## polynom  #########

def get_coords(img, scale=(256, 256)):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = cv.resize(gray_img, scale)
    coords = np.argwhere(gray_img < 50)
    coords = np.array([
        [x, scale[1] - y]
        for y, x in coords
    ])

    return coords


# получение матрицы расстояний для координат с заданной функцией для поиска расстояния
def distance_matrix(coords, distance_function):
    n = coords.shape[0]
    dm = np.zeros([n, n])
    for i, j in combinations(range(n), 2):
        dm[i, j] = distance_function(coords[i], coords[j])
        dm[j, i] = dm[i, j] # матрица симметричная

    for i in range(n):
        dm[i, i] = distance_function(coords[i], coords[i]) # диагональ может быть не нулевой
        
    return dm
def quasydiff2D_00(a,b):
    w_0=1
    w_1=0.1
    diff0=a[0]-b[0]
    diff1=a[1]-b[1]
    return np.array([w_0*diff0,w_1*diff1])


# получения функции расстояния с заданным значением epsilon
def gaussian_kernel_function(epsilon):
    def kernel_function(a, b):
        return np.exp(-(np.linalg.norm( quasydiff2D_00(a,b)) ** 2 / epsilon ** 2))
    return kernel_function


# матрица, в которой сумма строки == 1
def normalize(similarity_matrix):
    return np.apply_along_axis(lambda row: row / row.sum(), 1, similarity_matrix)


# итеративный поиск марковской цепи на шаге t
def markov_chain_on_iteration_slow(markov_chain, t):
    result = markov_chain
    for _ in range(1, t):
        np.dot(markov_chain, result)
    return result


# более эффективная реализация, основанная на том, что P**2k = P**k * P**k
def markov_chain_on_iteration(markov_chain, t):
    result = markov_chain
    i = 1
    while i * 2 <= t:
        result = np.dot(result, result)
        i *= 2
    
    for _ in range(i, t):
        result = np.dot(markov_chain, result)

    return result


# получение новых координат в диффузной карте, k - выходная размерность
def diffusion_map(markov_chain, k=3):
    eig_values, eig_vectors = np.linalg.eig(markov_chain)
    k_most = np.argsort(np.abs(eig_values))[-k:]
    return eig_vectors[:, k_most] * eig_values[k_most]



def plot_coords(coords):
    x_max, y_max = coords.max(axis=0)
    m = np.zeros((y_max + 1, x_max + 1))
    for x, y in coords:
        m[y][x] = 1

    plt.figure(figsize=(10,10))
    plt.imshow(m, cmap='binary', origin='lower')
    plt.show()


def plot_matrix(matrix):
    return plt.imshow(matrix, cmap='binary')


def plot_matrices(name_matrix_dict, height=6):
    n = len(name_matrix_dict)
    plt.figure(figsize=[height * n, height])
    for i, (name, matrix) in enumerate(name_matrix_dict.items()):
        plt.subplot(1, n, i + 1)
        plt.imshow(matrix, cmap='binary')
        plt.title(name)
    plt.show()

def represent_subclasses_plot_3D_00(subcl):
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot(projection='3d')
    colors_=['k','r','g','b','y','c','k','r','g','r','y','c']
    marker_=['2','<','1','<','<','<','*','*','*','o','o','o','o','*']

    subclasses=[]
    count=0
    for project_DM1 in subcl:
        cnt=np.mod(count,len(colors_))

        ax.scatter(project_DM1[:, -2], project_DM1[:, -3], project_DM1[:, -4] ,marker=marker_[cnt],  c=colors_[cnt],s=20, label= count)

        count+=1     
    plt.legend()
#########################
#########################

def plot_3d(coords):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
def kernel_motion_blur(  kernel_size, angle):
    '''
    param filter_size: integer, size of square blur filter.
    param angle: integer from range [-90,90]. clockwise angle between horizontal line and target line
        angle=0 means vertical blur
        angle=90 means horizontal blur
        angle>0 means downleft - upright blur
        angle<0 means upleft - downright blur
    returns: filter array of 0s and 1s of size (filter_size x filter_size) with normalization koefficient 1/filter_size
    '''
    if angle > 90:
        angle = 90
    elif angle < -90:
        angle = -90
    kernel_size = int(kernel_size)
    ab_angle = abs(angle)
    start_point = (0, 0)
    end_point = (kernel_size, np.int(kernel_size * np.tan(np.radians(min(ab_angle, 90 - ab_angle)))))
    kernel_motion = np.zeros((kernel_size, kernel_size))
    kernel_motion = cv2.line(kernel_motion, start_point, end_point, 1, 1)
    kernel_motion = kernel_motion / kernel_size
    if angle < -45:
        return np.flip(np.transpose(kernel_motion), 0)
    elif angle >= -45 and angle < 0:
        return np.flip(kernel_motion, 1)
    elif angle >= 0 and angle <= 45:
        return kernel_motion
    else:
        return np.transpose(kernel_motion)
 
def motion_blur(  imgarr):
    h, w, _ = imgarr.shape
    max_dim = h if h >= w else w
    min_ksize = 3 + round(max_dim / 128) if max_dim != 128 else 3  # для 128х128 3 - 5
    max_ksize = 5 + round(max_dim / 128) if max_dim != 128 else 5
    rand_ksize = random.randint(min_ksize, max_ksize)
    rand_angle = random.randint(-90, 90)
    kernel_mb = kernel_motion_blur(rand_ksize, rand_angle)
    imgarr = cv2.filter2D(imgarr, -1, kernel_mb)
 
    blur_ksize = round(rand_ksize // 3) if rand_ksize != 2 else 1  # для 128х128 1 - 2
    imgarr = cv2.blur(imgarr, (blur_ksize, blur_ksize), cv2.BORDER_DEFAULT)
    return imgarr

def get_coords_0(gray_img,img1):
    scale= gray_img.shape
     
    coords = np.argwhere(gray_img > 0.5)
     
    coords = np.array([
        [x, scale[1] - y,  *(20*img1[y, x,:])]
        for y, x in coords
    ])

    return coords
def get_coords_1(gray_img):
    scale= gray_img.shape
     
    coords = np.argwhere(gray_img > 0.5)
     
    coords = np.array([
        [ x, scale[1] - y  ]
        for y, x in coords
    ])

    return coords

def base_motion_kernels_00(n_,show):
    [x,y]=np.meshgrid(np.array(range(n_)),np.array(range(n_)))

    ker023=(x+y-n_+1)**2
    ker023=ker023-np.mean(ker023.ravel())
    ker023=ker023/np.linalg.norm(ker023)
    ker024=np.rot90(ker023)

    ker021=abs(x-(n_-1)/2)
    ker021=ker021-np.mean(ker021.ravel())
    ker021=ker021/np.linalg.norm(ker021)
    ker022=np.rot90(ker021)
    if show:
        ai_2(ker023)
        ai_2(ker024)

        ai_2(ker021)
        ai_2(ker022)
    return ker023,ker024,ker021,ker022

def get_coordinates_cluster_00(gray,eps_):

    gray1=1-gray.astype('float') /255
     
    ske =  skeletonize(gray1).astype('float') 
     
    coords= get_coords_1 (ske)



    clustering = DBSCAN(eps=eps_, min_samples=4).fit (coords) 
    cluster = clustering.labels_
    return ske,coords,cluster

def motion_clustering_00(n_,gray1,eps_  ):

    ker023,ker024,ker021,ker022=base_motion_kernels_00(n_,1)



    ske =  skeletonize(gray1).astype('float') 


    imgarr0 = np.expand_dims(cv2.filter2D(ske, -1, ker023.astype('float')),2)
    imgarr1 = np.expand_dims(cv2.filter2D(ske, -1, ker024.astype('float')),2)
    imgarr2 = np.expand_dims(cv2.filter2D(ske, -1, ker021.astype('float')),2)
    imgarr3 = np.expand_dims(cv2.filter2D(ske, -1, ker022.astype('float')),2)
    imgarr=np.concatenate([imgarr0,imgarr1,imgarr2,imgarr3],-1)

    coords_0 = get_coords_0 (ske,imgarr)


    clustering = DBSCAN(eps=eps_, min_samples=5).fit (coords_0) 
    cluster = clustering.labels_
    return cluster,ske,coords_0


def get_subspace_00(X_0):
    X_1=normalizat_str(X_0)
    X_2=normalizat_str(X_1**2)
    X_=np.array([0*X_0+1, X_1, X_2])
    return X_

def normalizat_str(X_0):
    X_1=X_0-np.mean(X_0)
    X_2=X_1/(max(X_1)-min(X_1))
    return np.array(X_2)

def lab2clusters_00(cluster,show):
    clusters=[]
    for i_ in list(set(cluster)) :
        ii_=np.where(cluster==i_)[0]
        if len(ii_)>30:
            clusters.append(coords[ii_,:])
            if show:
                print(i_, len(ii_))
                plt.figure()
                plt.cla()
                plt.plot(coords[:,0],coords[:,1],'.k')
                plt.plot(coords[np.where(cluster==i_)[0],0],coords[np.where(cluster==i_)[0],1],'*r')
                plt.show() 
    return clusters

def lab2clusters_01(cluster,show):
    clusters=[]
    for i_ in list(set(cluster)) :
        ii_=np.where(cluster==i_)[0]
        if len(ii_)>30:
            clusters.append([coords[ii_,:],ii_])
            if show:
                print(i_, len(ii_))
                plt.figure()
                plt.cla()
                plt.plot(coords[:,0],coords[:,1],'.k')
                plt.plot(coords[np.where(cluster==i_)[0],0],coords[np.where(cluster==i_)[0],1],'*r')
                plt.show() 
    return clusters

def polynomy_claster_00(coords,clusters,eps_,thr_,show):


    X_00=get_subspace_00(coords[:,1])
    Y_=coords[:,0]
    cluster_000=[]
    for cluster_0 in clusters:
        cluster_=cluster_0[0]
        ii_=cluster_0[1]
        X_0=cluster_[:,1]
        X_=X_00[:,ii_  ]


        _,coeff_09096=PrIntoSubsp_02(cluster_[:,0],X_ ,0.01)
        approximation1=np.dot(np.expand_dims(coeff_09096,0),X_00 )
        if show:
            plt.figure()
            plt.cla()
            plt.plot(coords[:,0],coords[:,1],'.k')
            plt.plot(cluster_[:,0],cluster_[:,1],'*r')
            plt.plot( approximation1[0,:],coords[:,1],'*g')
            plt.show() 

        for jhg in range(10):
            delta=(approximation1[0,:]-Y_)**2/eps_**2
            i_cluster=np.where(delta<thr_)[0]
            coord_claster=coords[i_cluster,:]
            clustering1 = DBSCAN(eps=3, min_samples=5).fit (coord_claster) 
            labels_ = clustering1.labels_
            i_cluster_0=[]
            for lab_0 in list(set(labels_)):
                if len(set(ii_)&set(i_cluster[labels_==lab_0]))>0:
                    i_cluster_0.append(i_cluster[labels_==lab_0])
            i_cluster_0=np.concatenate(i_cluster_0) 
            cluster_1=coords[i_cluster_0,:]
            X_=X_00[:,i_cluster_0  ]
            _,coeff_09096=PrIntoSubsp_02(cluster_1[:,0],X_ ,0.01)
            approximation1=np.dot(np.expand_dims(coeff_09096,0),X_00 )
            if show:
                plt.figure()
                plt.cla()
                plt.plot(coords[:,0],coords[:,1],'.k')
                plt.plot(cluster_[:,0],cluster_[:,1],'*r')
                plt.plot( approximation1[0,:],coords[:,1],'*g')
                plt.show() 
        if show:
            print('-------------------')
        cluster_000.append([cluster_1,i_cluster_0,coeff_09096])
    return cluster_000







