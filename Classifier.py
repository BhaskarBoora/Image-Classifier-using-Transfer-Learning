from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import backend as K
from tkinter import *
K.tensorflow_backend._get_available_gpus()
window=Tk()



def find():
    
    #img_file='h.jpg'
    #img_file='Test.jpg'
    
    img=cv2.imread(img_file.get(),cv2.IMREAD_COLOR)
    img=cv2.resize(img,(120,80))
    img1 = img[:,:]
    img1=np.array(img1,dtype=np.float64)
    img1=np.expand_dims(img1,axis=0)


    cnn='TL.h5'

    cnn=load_model(cnn)
    score = cnn.predict(img1)
    score=score[0]

    d={0:'organic',1:'recyclable',2:'electronic',3:'toxic'}

    ans=d[score.argmax()]
    t1.insert(END,ans)
    
b1=Button(window,text="Execute",command=find)
b1.grid(row=0,column=0)

img_file=StringVar()
e1=Entry(window,textvariable=img_file)
e1.grid(row=0,column=1)

t1=Text(window,height=1,width=20)
t1.grid(row=0,column=2)

window.mainloop()


