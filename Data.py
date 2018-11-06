import os
import cv2
import numpy as np
import pickle

def prepare_data(folder):
    data=[]
    
    for i in os.listdir(folder):
        img=cv2.imread(folder+'\\'+i,cv2.IMREAD_COLOR)
        img=cv2.resize(img,(120,80))
        img1 = img[:,:]
        img1=np.array(img1,dtype=np.float64)
        data.append(img1)
    return data


dir=os.getcwd()

O=dir+'\Organic'
R=dir+'\Recyclable'
E=dir+'\Electronic'
T=dir+'\Toxic'


o=prepare_data(O)
print('Done')
r=prepare_data(R)
e=prepare_data(E)
t=prepare_data(T)

data=[o,r,e,t]

save=open('Data.pkl','wb')
pickle.dump(data,save)
save.close() 