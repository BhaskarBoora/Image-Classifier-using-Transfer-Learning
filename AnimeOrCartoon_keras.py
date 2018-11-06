import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


from keras import backend as K
K.tensorflow_backend._get_available_gpus()



def load_data(file):
    file=open(file,'rb')
    data=pickle.load(file)
    file.close()
    return data

def process_data(d):
    O=d[0]
    R=d[1]
    E=d[2]
    T=d[3]
    
    o=len(O)
    r=len(R)
    e=len(E)
    t=len(T)
    
    m=o+r+e+t
    y=np.zeros(shape=(m,4))
    
    y[0:o,0]=1
    y[o:o+r,1]=1
    y[o+r:o+r+e,2]=1
    y[o+r+e:,3]=1
    
    X=[]
    for i in O:
        X.append(i)
    for i in R:
        X.append(i)
    for i in E:
        X.append(i)
    for i in T:
        X.append(i)
    
    
    X=np.array(X)
    return X,y


def model(X,y):
    net=Sequential()
    net.add(Conv2D(32,(5,5), activation='relu', input_shape=(80,120,3)))
    net.add(MaxPooling2D(pool_size=(3, 3)))
    net.add(Conv2D(64,(5,5), activation='relu'))
    net.add(Conv2D(64,(5,5), activation='relu'))
    net.add(MaxPooling2D(pool_size=(3, 3)))
    net.add(Flatten())
    net.add(Dense(512,activation='relu'))
    net.add(Dropout(0.5))
    net.add(Dense(4,activation='softmax'))
    
    opt = optimizers.SGD(lr=0.0001, momentum=0.8)
    net.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

    
    net.fit(X, y, batch_size=96, epochs=1)
    net.save('Custom.h5')
    
    return net

if __name__=='__main__':
    file='Data.pkl'
    data=load_data(file)
    print('Done')
    X,y=process_data(data)
    
    X,X_test,y,y_test=train_test_split(X,y,test_size=0.1)
    X_train,X_dev,y_train,y_dev=train_test_split(X,y,test_size=0.1)

    
    cnn=model(X_train,y_train)
    print('Dev= ',cnn.evaluate(X_dev, y_dev, batch_size=96))
    print('Test= ',cnn.evaluate(X_test, y_test, batch_size=96))
    
    
    