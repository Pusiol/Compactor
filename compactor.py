# -*- coding: utf-8 -*-
"""
@author: Delta
"""
from PIL import Image as img
#import math

import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy import signal




def decim(matrix, n):
    return matrix[::n,::n]

def uper(matrix, n):
    l=np.shape(matrix)[0]
    h=np.shape(matrix)[1]
    H=np.zeros((1,h*2))
    L=np.zeros((l,1))
    for i in range(h,0,-1):
        for j in range(0,n):
            matrix=np.concatenate((matrix[:,:i],L,matrix[:,i:]),1)

    for i in range(l,0,-1):
        for j in range(0,n):
            matrix=np.concatenate((matrix[:i,:],H,matrix[i:,:]),0)
    return matrix



def transform(matrix):
    #h0=(1+math.sqrt(3))/4/math.sqrt(2)
    #h1=(3+math.sqrt(3))/4/math.sqrt(2)
    #h2=(3-math.sqrt(3))/4/math.sqrt(2)
    #h3=(1-math.sqrt(3))/4/math.sqrt(2)
    
    #w=np.asarray([[h0,h1,h2,h3],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    w=np.asarray([[3.9347319995026124e-05, -0.0002519631889981789, 0.00023038576399541288, 0.0018476468829611268, -0.004281503681904723, -0.004723204757894831, 0.022361662123515244, 0.00025094711499193845, -0.06763282905952399, 0.030725681478322865, 0.14854074933476008, -0.09684078322087904, -0.29327378327258685, 0.13319738582208895, 0.6572880780366389, 0.6048231236767786, 0.24383467463766728, 0.03807794736316728],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    ww=np.transpose(w)
    
    #s=np.asarray([[h3,-h2,h1,-h0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    s=np.asarray([[-0.03807794736316728, 0.24383467463766728, -0.6048231236767786, 0.6572880780366389, -0.13319738582208895, -0.29327378327258685, 0.09684078322087904, 0.14854074933476008, -0.030725681478322865, -0.06763282905952399, -0.00025094711499193845, 0.022361662123515244, 0.004723204757894831, -0.004281503681904723, -0.0018476468829611268, 0.00023038576399541288, 0.0002519631889981789, 3.9347319995026124e-05],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    ss=np.transpose(s)
    
    qb1=signal.convolve2d(matrix,  w, mode='full', boundary='symm')
    qb1=signal.convolve2d(qb1, ww , mode='full', boundary='symm')
    qb1=decim(qb1,2)#[sl:,sl:]
    
    qb2=signal.convolve2d(matrix, s, mode='full', boundary='symm')
    qb2=signal.convolve2d(qb2, ww , mode='full', boundary='symm')
    qb2=decim(qb2,2)#[sl:,sl:]#qb2=uper(h,l,qb2,1)
    
    qb3=signal.convolve2d(matrix, w, mode='full', boundary='symm')
    qb3=signal.convolve2d(qb3, ss , mode='full', boundary='symm')
    qb3=decim(qb3,2)#[sl:,sl:]
    
    qb4=signal.convolve2d(matrix, s, mode='full', boundary='symm')
    qb4=signal.convolve2d(qb4, ss , mode='full', boundary='symm')
    qb4=decim(qb4,2)#[sl:,sl:]

    return qb1,qb2,qb3,qb4
    
    
    
    
def reverse(m1,m2,m3,m4):
    A1=uper(m1,1)
    A2=uper(m2,1)
    A3=uper(m3,1)
    A4=uper(m4,1)
    
    #W=np.asarray([[0,0,0,0],[0,0,0,0],[0,0,0,0],[h2,h1,h0,h3]])
    W=np.asarray([[0.03807794736316728, 0.24383467463766728, 0.6048231236767786, 0.6572880780366389, 0.13319738582208895, -0.29327378327258685, -0.09684078322087904, 0.14854074933476008, 0.030725681478322865, -0.06763282905952399, 0.00025094711499193845, 0.022361662123515244, -0.004723204757894831, -0.004281503681904723, 0.0018476468829611268, 0.00023038576399541288, -0.0002519631889981789, 3.9347319995026124e-05],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    WW=np.transpose(W)
    
    #S=np.asarray([[0,0,0,0],[0,0,0,0],[0,0,0,0],[h3,-h0,h1,-h2]])
    S=np.asarray([[3.9347319995026124e-05, 0.0002519631889981789, 0.00023038576399541288, -0.0018476468829611268, -0.004281503681904723, 0.004723204757894831, 0.022361662123515244, -0.00025094711499193845, -0.06763282905952399, -0.030725681478322865, 0.14854074933476008, 0.09684078322087904, -0.29327378327258685, -0.13319738582208895, 0.6572880780366389, -0.6048231236767786, 0.24383467463766728, -0.03807794736316728],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    SS=np.transpose(S)
    
    
    A1=signal.convolve2d(A1, WW, mode='full', boundary='symm') # symm
    A1=signal.convolve2d(A1, W, mode='full', boundary='symm')
    
    A2=signal.convolve2d(A2, WW, mode='full', boundary='symm')
    A2=signal.convolve2d(A2, S, mode='full', boundary='symm')
    
    A3=signal.convolve2d(A3, SS, mode='full', boundary='symm')
    A3=signal.convolve2d(A3, W, mode='full', boundary='symm')
    
    A4=signal.convolve2d(A4, SS, mode='full', boundary='symm')
    A4=signal.convolve2d(A4, S, mode='full', boundary='symm')
    
    A = (A1+A2+A3+A4)
    
    A=A[:A.shape[0]-23,:A.shape[1]-23]
    A=A[17:,17:]

    return A
    
    
    
    
def empacota(qb2,qb3,qb4,n2,n3):
    fac=[np.max(qb2),np.max(qb3),np.max(qb4)]
    fic=[np.min(qb2),np.min(qb3),np.min(qb4)]
    
    wqb2=np.uint8(  (qb2-fic[0])/(fac[0]-fic[0])*np.power(2,n2)  ).reshape((1,qb2.size))
    wqb3=np.uint8(  (qb3-fic[1])/(fac[1]-fic[1])*np.power(2,n2)  ).reshape((1,qb2.size))
    wqb4=np.uint8(  (qb4-fic[2])/(fac[2]-fic[2])*np.power(2,n3)  ).reshape((1,qb2.size))
    
    return np.concatenate(([[fac[0],fic[0]]],wqb2,[[fac[1],fic[1]]],wqb3,[[fac[2],fic[2]]],wqb4),1)

    
def empacotaum(qb1,n1):
    fac=np.max(qb1)
    fic=np.min(qb1)
    
    wqb1=np.uint8(  (qb1-fic)/(fac-fic)*np.power(2,n1)  ).reshape((1,qb1.size))
    
    return np.concatenate(([[fac,fic]],wqb1),1)
    
    
    
    
if __name__ == "__main__":
    l=img.open(sys.argv[1])
    margin=0
    tmax=0
    
    m=l.convert("YCbCr", 0, 256)
    Y,B,R = m.split()
    
    plt.imshow(m)
    plt.figure()
    
    h=np.size(m)[0]+20
    l=np.size(m)[1]+20
    
    y=np.asarray(Y.getdata()).reshape(m.size[::-1])
    b=np.asarray(B.getdata()).reshape(m.size[::-1])
    r=np.asarray(R.getdata()).reshape(m.size[::-1])
    
    
    y1,y2,y3,y4=transform(y)
    b1,b2,b3,b4=transform(b)
    r1,r2,r3,r4=transform(r)
    
    y11,y22,y33,y44=transform(y1)
    b11,b22,b33,b44=transform(b1)
    r11,r22,r33,r44=transform(r1)
    
    yy2=y2-np.min(y2)
    yy3=y3-np.min(y3)
    yy4=y4-np.min(y4)
    QB1 = img.fromarray(np.uint8(y1*255/np.max(y1)))
    QB2 = img.fromarray(np.uint8(yy2*255/np.max(yy2)))
    QB3 = img.fromarray(np.uint8(yy3*255/np.max(yy3)))
    QB4 = img.fromarray(np.uint8(yy4*255/np.max(yy4)))
    
    
    plt.subplot(221)
    plt.imshow(QB1,cmap='gray')
    
    #im = img.merge("RGB", (QB, QB, QB))
    plt.subplot(222)
    plt.imshow(QB2,cmap='gray')
    
    plt.subplot(223)
    plt.imshow(QB3,cmap='gray')
    
    plt.subplot(224)
    plt.imshow(QB4,cmap='gray')
    
    
    qb =np.concatenate((empacota(y2,y3,y4,4,3),empacota(b2,b3,b4,3,2),empacota(r2,r3,r4,3,2)),1)
    qbs=np.concatenate((empacotaum(y1,8),empacotaum(b1,4),empacotaum(r1,4),[y1.shape]),1)
    
    np.savez_compressed('uva',s=qb,t=qbs)#,allow_pickle='false')