import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import xlwt
from numba import jit



np.set_printoptions(suppress=True)


def letterbox_image(img, img_mask):
    ih, iw,ch = img.shape
    h, w, ch =img_mask.shape
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    img = cv2.resize(img, (nw,nh), interpolation = cv2.INTER_AREA)
    new_img = cv2.copyMakeBorder(img, int((h-nh)/2), int((h-nh)/2), int((w-nw)/2), int((w-nw)/2), cv2.BORDER_CONSTANT, value=(0,0,0))   

    return new_img

def un_letterbox_image(img, img_mask):
    ih, iw,ch = img_mask.shape
    h, w, ch = img.shape

    if h/ih > w/iw:       
        nw = int(w/(h/ih))
        img_mask = img_mask[:, int((iw-nw)/2):int((iw-nw)/2+nw)]
    else :
        nh = int(h/(w/iw))
        img_mask = img_mask[int((ih-nh)/2):int((ih-nh)/2+nh),:]
 
    img_mask = cv2.resize(img_mask, (w,h), interpolation = cv2.INTER_AREA)

    return img_mask   


def find_color(img,img_mask):
    point_number = np.sum(img_mask)/255
    b,g,r = np.array(cv2.split(img))
    b_mean = np.sum(b)/point_number
    g_mean = np.sum(g)/point_number
    r_mean = np.sum(r)/point_number



    if b_mean !=  0 and g_mean !=  0 and  r_mean !=  0 :
        bdg = b_mean/g_mean
        bdr = b_mean/r_mean
        gdr = g_mean/r_mean

    return b_mean,g_mean,r_mean,bdg,bdr,gdr


def find_shape(img_gray,standard_long,standard_area):

    
    contours, hierarchy = cv2.findContours(img_gray,  # 寻找轮廓的图像
                                           cv2.RETR_EXTERNAL,  # 检索模式
                                           cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours,key=cv2.contourArea)
    
    length = cv2.arcLength(contours[-1], True)
    length = length*standard_long
    area_projection = np.sum(img_gray)/255
    area_profile = cv2.contourArea(contours[-1])
    area_projection = area_projection*standard_area
    area_profile = area_profile*standard_area



    max_contour = contours[-1]

    hull_points = cv2.convexHull(max_contour )   #凸包  
    hull_points_number = hull_points.shape[0]

    hull_area = cv2.contourArea(hull_points)
    hull_area = hull_area*standard_area



    # 找面积最小的矩形
    rect = cv2.minAreaRect(max_contour)
    w,h = rect[1]
    # 得到最小矩形的坐标
    box = cv2.boxPoints(rect)
    # 标准化坐标到整数
    box = np.int0(box)

    w = w*standard_long
    h = h*standard_long

    (x_c,y_c),radius = cv2.minEnclosingCircle(max_contour)

    

    rect_area = w*h
    radius = radius*standard_long
    out_circle_area = radius*radius*3.1415926535

    if w>h:
        shape_CI = 1.273*area_projection/w
    else:
        shape_CI = 1.273*area_projection/h

    circle_CI = 12.5663706143591729538504*area_projection/(length**2)
    out_circle_CI = area_projection/out_circle_area

    TBA = area_projection/rect_area   #投影面积与外接矩形面积之比
    HWR = h/w   #高宽之比
    PAR = length/area_projection


    return length,area_profile,w,h,radius,rect_area,out_circle_area,shape_CI,circle_CI,out_circle_CI,TBA,HWR,PAR,hull_area,hull_points_number,box

def glgcm(img_gray, ngrad=16, ngray=16):
    '''Gray Level-Gradient Co-occurrence Matrix,取归一化后的灰度值、梯度值分别为16、16'''
    # 利用sobel算子分别计算x-y方向上的梯度值
    gsx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    height, width = img_gray.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5 # 计算梯度值
    grad = np.asarray(1.0 * grad * (ngrad-1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_gray * (ngray-1) / img_gray.max(), dtype=np.int16) # 0-255变换为0-15
    gray_grad = np.zeros([ngray, ngrad]) # 灰度梯度共生矩阵

    gray_grad = get_gray_grad(height, width, gray, grad, gray_grad)
    gray_grad = 1.0 * gray_grad / (height * width) # 归一化灰度梯度矩阵，减少计算量
    glgcm_features = get_glgcm_features(gray_grad)
    return glgcm_features

@jit
def get_gray_grad(height, width, gray, grad, gray_grad):
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    return gray_grad
    
@jit
def get_glgcm_features(mat):
    '''根据灰度梯度共生矩阵计算纹理特征量，包括小梯度优势，大梯度优势，灰度分布不均匀性，梯度分布不均匀性，能量，灰度平均，梯度平均，
    灰度方差，梯度方差，相关，灰度熵，梯度熵，混合熵，惯性，逆差矩'''
    sum_mat = mat.sum()
    small_grads_dominance = big_grads_dominance = gray_asymmetry = grads_asymmetry = energy = gray_mean = grads_mean = 0
    gray_variance = grads_variance = corelation = gray_entropy = grads_entropy = entropy = inertia = differ_moment = 0
    for i in range(mat.shape[0]):
        gray_variance_temp = 0
        for j in range(mat.shape[1]):
            small_grads_dominance += mat[i][j] / ((j + 1) ** 2)
            big_grads_dominance += mat[i][j] * j ** 2
            energy += mat[i][j] ** 2
            if mat[i].sum() != 0:
                gray_entropy -= mat[i][j] * np.log(mat[i].sum())
            if mat[:, j].sum() != 0:
                grads_entropy -= mat[i][j] * np.log(mat[:, j].sum())
            if mat[i][j] != 0:
                entropy -= mat[i][j] * np.log(mat[i][j])
                inertia += (i - j) ** 2 * np.log(mat[i][j])
            differ_moment += mat[i][j] / (1 + (i - j) ** 2)
            gray_variance_temp += mat[i][j] ** 0.5

        gray_asymmetry += mat[i].sum() ** 2
        gray_mean += i * mat[i].sum() ** 2
        gray_variance += (i - gray_mean) ** 2 * gray_variance_temp
    for j in range(mat.shape[1]):
        grads_variance_temp = 0
        for i in range(mat.shape[0]):
            grads_variance_temp += mat[i][j] ** 0.5
        grads_asymmetry += mat[:, j].sum() ** 2
        grads_mean += j * mat[:, j].sum() ** 2
        grads_variance += (j - grads_mean) ** 2 * grads_variance_temp
    small_grads_dominance /= sum_mat
    big_grads_dominance /= sum_mat
    gray_asymmetry /= sum_mat
    grads_asymmetry /= sum_mat
    gray_variance = gray_variance ** 0.5
    grads_variance = grads_variance ** 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            corelation += (i - gray_mean) * (j - grads_mean) * mat[i][j]
    return small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean, grads_mean,gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia, differ_moment
@jit
def histogram(hist):
    hist_std = np.std(hist)
    normal_hist = hist/hist.sum()
    I_entropy_total = 0
    for hi in range(0,256):
        if normal_hist[hi][0] != 0:
            I_entropy = -(normal_hist[hi][0] * np.log(normal_hist[hi][0])/np.log(2))
            I_entropy_total +=I_entropy
    return hist_std,I_entropy_total


 



 


standard_long = 1
standard_area = 1

def phenotype(filePath):

    head = ['处理','重复','编号',\
            '蓝色均值','绿色均值','红色均值','蓝绿比值','蓝红比值','绿红比值','周长','轮廓面积','外接矩形宽','外接矩形高','外接圆半径','外接矩形面积','外接圆面积',\
            '形状率紧凑度','圆形率紧凑度','外接圆紧凑度','投影面积与外接矩形面积之比','高宽之比','周长与面积之比','凸包面积','凸包顶点数',\
            '小梯度优势','大梯度优势','灰度分布不均匀性','梯度分布不均匀性','能量','灰度平均','梯度平均','灰度方差','梯度方差','相关','灰度熵','梯度熵','混合熵','惯性','逆差矩','灰度直方图方差','灰度直方图熵']
    statistics = xlwt.Workbook(encoding = 'utf-8')
    sheet = statistics.add_sheet(sheetname='豆荚表型参数')
    for i in range(len(head)):
        sheet.write(0,i,head[i])
    k = 1
    i = 0
    for manAge in os.listdir(filePath):        
        for rePetition in os.listdir(filePath + '/' + manAge):              
            for fileName in os.listdir(filePath + '/' + manAge + '/' + rePetition):
                lst = []
                numBer = fileName.split('-')[2]

                sheet.write(k,1,rePetition)
                sheet.write(k,0,manAge)  
                sheet.write(k,2,numBer) 


                img = cv2.imread(filePath + '/' + manAge + '/' + rePetition + '/' + fileName)
                img_PIL = Image.open(filePath + '/' + manAge + '/' + rePetition + '/' + fileName)


                img_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, img_mask = cv2.threshold(img_mask,10, 255, cv2.THRESH_BINARY)



                #将RGB转为HSI
                img_HSI = np.array(img, dtype=np.float32) / 3
                (b, g, r) = cv2.split(img_HSI)
                I = g + b + r
                I = np.array(I, dtype=np.uint8)

                

                img_cut_color = cv2.bitwise_and(img,img,mask = img_mask)
                I_cut = cv2.bitwise_and(I,I,mask = img_mask)
                
                b_mean,g_mean,r_mean,bdg,bdr,gdr = find_color(img_cut_color,img_mask)


                length,area_profile,w,h,radius,rect_area,out_circle_area,shape_CI,circle_CI,out_circle_CI,TBA,HWR,PAR,hull_area,hull_points_number,box = find_shape(img_mask,standard_long,standard_area) 



                small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean, grads_mean,gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia, differ_moment = glgcm(I_cut, 15, 15)

                hist = cv2.calcHist([I],[0],None,[256],[0,255])
                hist_std,I_entropy_total = histogram(hist)
                
                lst += [b_mean,g_mean,r_mean,bdg,bdr,gdr]
                lst += [length,area_profile,w,h,radius,rect_area,out_circle_area,shape_CI,circle_CI,out_circle_CI,TBA,HWR,PAR,hull_area,hull_points_number]
                lst += [small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean, grads_mean,gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia, differ_moment]
                lst += [hist_std,I_entropy_total]
                lst = np.array(lst) 

                for v in range(len(lst)):
                    sheet.write(k,v+3,lst[v])


                k += 1



    statistics.save('phenotype' + '.xls')
