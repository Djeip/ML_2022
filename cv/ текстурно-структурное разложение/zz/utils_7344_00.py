import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_im_2(img, img2):
    """
    Визуальное представление двух имеджей в фигуре 
    На вход подаются следующие параметры:
    :param img - имейдж номер 1
    :param img2 - имейдж номер 2
    """
    plt.figure(figsize=(25,10))

    plt.plot(img.ravel() ,'k')
    plt.plot(img2.ravel() ,'r')
     
    plt.show()  

def plot_im_3(img, img2, img3):
    """
    Визуальное представление трех имеджей в фигуре
    На вход подаются следующие параметры:
    :param img - имейдж номер 1
    :param img2 - имейдж номер 2
    :param img3 - имейдж номер 3 
    """    
    plt.figure(figsize=(10,5))

    plt.plot(img.ravel() ,'y')
    plt.plot(img2.ravel() ,'k')
    plt.plot(img3.ravel() ,'r')
     
    plt.show()  

def show_img_1(img,cmap=None):
    """
    Визуальное представление одного имейджа в фигуре
    На вход подаются следующие параметры:
    :param img - имейдж
    :param cmap - Экземпляр карты цветов
    """        
    plt.figure()
    plt.imshow(img,cmap)
    plt.show()

def ai_2(img):
    """
    Визуальное представление имейджа в диапазоне значений от 0 до 1 в сером цвете
    На вход подаются следующие параметры:
    :param img - имейдж
    """            
    mmin = np.min(img)
    mmax = np.max(img)
    plt.imshow(255-255*(img-mmin)/(mmax-mmin), cmap = 'Greys')
    plt.show()
    return

def plot_im(img):
    """
    Визуальное представление одного имейджа.
    На вход подаются следующие параметры:
    :param img - имейдж
    """     
    Y_vect=img.ravel()
    plt.figure(figsize=(25,10))

    plt.plot(Y_vect ,'k')
     
    plt.show() 


def morph_2D_0(img,n):
    """
    Функция для расширения изображения и его размытия
    :param img имейдж
    :param n - размер ядра которым проходятся по имейджу
    """  
    kernel = np.ones((n, n), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    erose_img = cv2.erode(dilate_img, kernel, iterations=1)
    return erose_img

def IOU_mask_01(a_predict,b_true, thr):
    """
    Функция для выявления меры IOU качества сегментаци
    На вход подаются следующие параметры:
    :param thr порог значимости сегмента (в пикселях)
    :param a_predict, b_true - два тензора размера m на n которые сравниваем. m и n- размер изобрражения
    Тензоры в пикселях имеют 0 когда фон и 1 когда объект. значения между 0 и 1 квантизируются порогом 0.5
    """   
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
