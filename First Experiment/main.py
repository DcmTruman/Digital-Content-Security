from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
import random
import scipy.ndimage as ndimage
import scipy.signal as signal

#椒盐噪声的比例
salt_n = 0.02

#高斯函数标准差
gauss_sigma = 25

#滤波函数的list

filter_list = [signal.medfilt,signal.wiener,ndimage.gaussian_filter]
MED_FILTER = 0
WIENER_FILTER = 1
GAUSS_FILTER = 2

#滤波的size
filter_n = 3

#高斯平滑滤波的sigma
filter_sigma = 1

#图片名（当前py文件路径）
image_path = "lena_color.jpg"


def salt_pepper(img2 , n):
        """
        随机的选择 1/2*n*像素个数 的点，置黑
        随机的选择 1/2*n*像素个数 的点，置白
        """
        img = img2.copy()
        n = 1 if n > 1 else n
        row , colum , dims = img.shape
        pixel_num = int(row * colum * n * 0.5)
        for i in range(pixel_num):
                pos_i = np.random.randint(0 , row)
                pos_j = np.random.randint(0 , colum)
                pos_m = np.random.randint(0 , row)
                pos_n = np.random.randint(0 , colum)
                img[pos_i,pos_j,:] = 255
                img[pos_m,pos_n,:] = 0
        return img


def gauss_noise(img2 , n):
        """
        根据标准差生产符合高斯分布的噪声，加在原来的像素值上
        如果大于255，置为255
        如果小于0，置为0
        """
        img = img2.copy()
        row , colum , dims = img.shape
        for i in range(row):
                for j in range(colum):
                        for k in range(dims):
                                tmp = img[i][j][k] + random.gauss(0,n)
                                img[i][j][k] = 255 if tmp > 255 else 0 if tmp < 0 else tmp
        return img
        
def img_filter(img2 , n , func_type):
        """
        
        """
        img = img2.copy()
        row , colum , dims = img.shape
        for i in range(dims):
                img[:,:,i] = filter_list[func_type](img[:,:,i], n)
        
        return img

def show_gray_hist(img2):
        img =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)       
        plt.hist(img.ravel(),256,[0,256])
        plt.show()

        



if __name__ == '__main__' :
        img = cv2.imread(image_path)

#-------------------------------------------------------------------
        salt_pepper_img = salt_pepper(img , salt_n)
        gauss_noise_img = gauss_noise(img , gauss_sigma)
        #cv2.imshow('salt_pepper_image' , salt_pepper_img)
        #cv2.imshow('gauss_noise_img' , gauss_noise_img)
##-------------------------------------------------------------------
        med_filter_gauss_img = img_filter(gauss_noise_img , filter_n , MED_FILTER)
        med_filter_salt_img = img_filter(salt_pepper_img , 5 , MED_FILTER)

        #cv2.imshow('med_filter_gauss_img' , med_filter_gauss_img)
        #cv2.imshow('med_filter_salt_img' , med_filter_salt_img)
##-------------------------------------------------------------------
        #wiener_filter_gauss_img = img_filter(gauss_noise_img , filter_n , WIENER_FILTER)
        #wiener_filter_salt_img = img_filter(salt_pepper_img , 5 , WIENER_FILTER)

        #cv2.imshow('wiener_filter_gauss_img' , wiener_filter_gauss_img)
        #cv2.imshow('wiener_filter_salt_img' , wiener_filter_salt_img)

##--------------------------------------------------------------------
        gauss_filter_gauss_img = img_filter(gauss_noise_img , filter_sigma , GAUSS_FILTER)
        gauss_filter_salt_img = img_filter(salt_pepper_img , 5 , GAUSS_FILTER)

        #cv2.imwrite(".\\images\\gauss_filter_gauss_img.jpg",gauss_filter_gauss_img)
        #cv2.imwrite(".\\images\\gauss_filter_salt_img.jpg",gauss_filter_salt_img)
        #cv2.imshow('gauss_filter_gauss_img' , gauss_filter_gauss_img)
        #cv2.imshow('gauss_filter_salt_img' , gauss_filter_salt_img)
#--------------------------------------------------------------------
        show_gray_hist(gauss_noise_img)
        show_gray_hist(gauss_filter_gauss_img)
         
        cv2.waitKey(0)
        cv2.destroyAllWindows()

