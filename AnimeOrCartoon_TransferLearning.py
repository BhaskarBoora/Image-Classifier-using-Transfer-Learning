from keras import applications
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

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

file='Data.pkl'
data=load_data(file)
X,y=process_data(data)

print('Data loaded...')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
#X_train,X_dev,y_train,y_dev=train_test_split(X,y,test_size=0.1)

img_width=80
img_height=120

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers[:5]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(4, activation="softmax")(x)

model = Model(inputs = model.input, outputs = predictions)
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=96, epochs=50)
#dev_score = model.evaluate(X_dev, y_dev, batch_size=96)
test_score = model.evaluate(X_test, y_test, batch_size=96)
model.save("TL.h5")

print(test_score)

'''

Training 99.9
Dev 90
Test 92

'''