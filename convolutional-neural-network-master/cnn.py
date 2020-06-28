# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:54:02 2020

@author: @aadarshcodes
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()
model.add(Convolution2D(64,3,3,input_shape = (128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dense(output_dim=1,activation='sigmoid'))
model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')

model.fit(
        train_generator,
        steps_per_epoch=250,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=2000
        )

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cat8.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict_classes(test_image)
print(result)
train_generator.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print("The image is of a ",prediction)
else:
    prediction = 'cat'
    print("The image is of a ",prediction)