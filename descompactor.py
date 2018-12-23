# -*- coding: utf-8 -*-
"""
@author: Delta
"""
from PIL import Image as img
#import math

import matplotlib.pyplot as plt
import numpy as np
#import sys

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







def empacota(qb1,qb2,qb3,qb4,n1,n2,n3):
    fac=[np.max(qb1),np.max(qb2),np.max(qb3),np.max(qb4)]
    fic=[np.min(qb1),np.min(qb1),np.min(qb1),np.min(qb1)]
    
    wqb1=np.uint8(  (qb1-fic[0])/(fac[0]-fic[0])*np.power(2,n1)  ).reshape((1,qb1.size))
    wqb2=np.uint8(  (qb2-fic[1])/(fac[1]-fic[1])*np.power(2,n2)  ).reshape((1,qb1.size))
    wqb3=np.uint8(  (qb3-fic[2])/(fac[2]-fic[2])*np.power(2,n2)  ).reshape((1,qb1.size))
    wqb4=np.uint8(  (qb4-fic[3])/(fac[3]-fic[3])*np.power(2,n3)  ).reshape((1,qb1.size))
    
    return np.concatenate(([[fac[0],fic[0]]],wqb1,[[fac[1],fic[1]]],wqb2,[[fac[2],fic[2]]],wqb3,[[fac[3],fic[3]]],wqb4),1)

    
def desempacota(amst,vect,hhh,lll,n1,n2,n3):
    lot=amst.size
    #qba=np.empty()
    #for i in range(1,4):
    #    qba[i]=vect[0,2+lot*0:lot*1].reshape(lll,hhh)
    lll=int(lll)
    hhh=int(hhh)

    qba=amst[2+lot*0:lot*1].reshape(lll,hhh)
    qbb=vect[2+lot*0:lot*1].reshape(lll,hhh)
    qbc=vect[2+lot*1:lot*2].reshape(lll,hhh)
    qbd=vect[2+lot*2:lot*3].reshape(lll,hhh)
    
    qba= (qba)*(amst[0]-amst[1])          /np.power(2,n1)+amst[1]
    qbb= (qbb)*(vect[0]-vect[1])          /np.power(2,n2)+vect[1]
    qbc= (qbc)*(vect[lot*1]-vect[1+lot*1])/np.power(2,n2)+vect[1+lot*1]
    qbd= (qbd)*(vect[lot*2]-vect[1+lot*2])/np.power(2,n3)+vect[1+lot*2]    
    
    return qba,qbb,qbc,qbd


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



    

a=np.load('uva.npz')
qb =a['s']
qbs=a['t']

lot=int((qb.size)/3)
loti=int((qbs.size-2)/3)
hhh=qbs[0,qbs.size-1]
lll=qbs[0,qbs.size-2]

y1,y2,y3,y4=desempacota(qbs[0,loti*0:loti*1],qb[0,lot*0:lot*1],hhh,lll,8,4,3)
b1,b2,b3,b4=desempacota(qbs[0,loti*1:loti*2],qb[0,lot*1:lot*2],hhh,lll,4,3,2)
r1,r2,r3,r4=desempacota(qbs[0,loti*2:loti*3],qb[0,lot*2:lot*3],hhh,lll,4,3,2)


    
    
    
#    A=reverse(qba+fic1,qbb+fic2,qbc+fic3,qbd+fic4)
Y=reverse(y1,y2,y3,y4)
B=reverse(b1,b2,b3,b4)
R=reverse(r1,r2,r3,r4)




#    #A=reverse(np.uint8(qb1/fac1*255)*fac1/255,np.uint8(qb2/fac2*255)*fac2/255,np.uint8(qb3/fac3*255)*fac3/255,np.uint8(qb4/fac4*255)*fac4/255)
#    
#    #A=reverse(np.uint8(qb1/fac1*255)*fac1/255,qb2,qb3,np.uint8(qb4/fac4*255)*fac4/255)
#    A=reverse(qb1,np.uint16(qb2/fac2*65000)*fac2/65000,np.uint16(qb3/fac3*65000)*fac3/65000,qb4)
    
YY = img.fromarray(np.uint8(Y))
BB = img.fromarray(np.uint8(B))
RR = img.fromarray(np.uint8(R))
im = img.merge("YCbCr", (YY, BB, RR))

m=im.convert("RGB", 0, 256)
plt.imshow(m)    
    
m.save("tc.bmp")
    





plt.show()


