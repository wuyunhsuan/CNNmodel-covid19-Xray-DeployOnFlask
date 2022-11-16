"""
reference
Kaggle Covid-19 Image Dataset
https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
The original dataset seperates pics to train/test and has 3 different labels(covid/normal/Viral Pneumonia).
However, I combined train set with test set, and split data into new train/test set randomly(with sklearn.model_selection.train_test_split).
"""
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import os.path as path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


IMAGEPATH = 'Covid19'
imageW=64
imageH=64
imageC=3
dimX=4   # dimX=1-->MLP/ dimX=4-->CNN

# load files in 'IMAGEPATH'
dirs = os.listdir(IMAGEPATH)
X=[]
Y=[]
i=0
for files in dirs:
    file_paths = glob.glob(path.join(IMAGEPATH+"/"+files, '*.*'))
    for file in file_paths:
        img = cv2.imread(file)
        img = cv2.resize(img, (imageW, imageH), interpolation=cv2.INTER_AREA)
        X.append(img)
        # Y-->label
        Y.append(i)

    i=i+1

# normalize input data
X = np.asarray(X)
Y = np.asarray(Y)
X = X.reshape(X.shape[0],imageW,imageH,imageC)
X = X.astype('float32')
X = X/255

dim = X.shape[1]
category = len(dirs)
# split data
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.05)
# One-hot encoding
train_y = tf.keras.utils.to_categorical(train_y, category)
test_y = tf.keras.utils.to_categorical(test_y, category)

# build CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,
                                 kernel_size=(3,3),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(imageW,imageH,imageC)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))
model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256,activation="relu"))
model.add(tf.keras.layers.Dense(units=256,activation="relu"))
model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))
model.compile(optimizer="adam",
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.summary()

# save model
with open("covid_model_git.json", "w") as json_file:
   json_file.write(model.to_json())
# save weights
model.save_weights("covid_model_git.h5")

# save weights
try:
    with open('covid_model_git.h5', 'r') as load_weights:
        model.load_weights("covid_model_git.h5")
except IOError:
    print("File not exists")

checkpoint = tf.keras.callbacks.ModelCheckpoint("covid_model_git.h5", monitor='loss', verbose=1,
                                                save_best_only=True, mode='auto', save_freq=50)

# train
model.fit(train_x,train_y,epochs=100,callbacks=[checkpoint])

# test
score = model.evaluate(test_x, test_y)
print("loss:",score[0])
print("accuracy:",score[1])
predict = model.predict(test_x)
print("Ans:",np.argmax(predict[0]),",","Prediction:",dirs[np.argmax(predict[0])])

nrow=3
ncol=3
fig, axs = plt.subplots(nrows=nrow,ncols=ncol, figsize=(nrow,ncol))
fig.subplots_adjust(hspace=0.3, wspace=0.1)
for row in range(nrow):
    for col in range(ncol):
        i=(row*ncol)+col
        pic=test_x[i].reshape(imageW,imageH,imageC)
        axs[row,col].imshow(pic)
        ans=np.argmax(predict[i], axis=-1)
        pic_ans="Prediction:"+str(ans)+ " , "+str(dirs[ans])
        axs[row,col].set_title(pic_ans)

plt.savefig('Covid-CNN.png')
plt.show()
