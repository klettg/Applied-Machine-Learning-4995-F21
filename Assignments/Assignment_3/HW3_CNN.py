#!/usr/bin/env python
# coding: utf-8

# # Homework 3: Convolutional Neural Networks
# 
# Due Wednesday 11/24 at 11:59 pm EST

# Download the dataset `cats-notcats` from github (given as a part of the assignment). This dataset has images of cats and images that are not cats (in separate folders). The task is to train a convolutional neural network (CNN) to build a classifier that can classify a new image as either `cat` or `not cat`

# 1. Load the dataset and create three stratified splits - train/validation/test in the ratio of 70/10/20. 

# In[4]:


import tensorflow as tf
from tensorflow import keras

batch_size=32
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
all_data_batches = image_generator.flow_from_directory(directory='/Users/Griffin/repos/assignment-3-klettg/data/cats-notcats', 
                                                       class_mode='binary', target_size=(256, 256), batch_size=batch_size)

val_gen = image_generator.flow_from_directory(directory='/Users/Griffin/repos/assignment-3-klettg/data/cats-notcats', subset='validation',
                                              class_mode='binary', target_size=(256, 256), batch_size=batch_size)




#print(batches)


# In[5]:


#for batch in all_data_batches:
#    print(batch)


# 2. Create a CNN that has the following hidden layers:
# 
#     a. 2D convolution layer with a 3x3 kernel size, has 128 filters, stride of 1 and padded to yield the same size as input, followed by a ReLU activation layer
#     
#     b. Max pooling layer of 2x2
#     
#     c. Dense layer with 128 dimensions and ReLU as the activation layer

# In[6]:


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', input_shape=(256,256,3)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", name="c"),
    ]
)
model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])


# 3. Train the classifier for 20 epochs with 100 steps per epoch. Also use the validation data during training the estimator.

# In[7]:


model.fit_generator(all_data_batches, epochs=20, steps_per_epoch=100, validation_data=val_gen)
model.save_weights(first_try.h5)


# 4. Plot the accuracy and the loss over epochs for train & validation sets

# In[ ]:





# 5. Add the following layers to (2) before the dense layer:
# 
#     a. 2D convolution layer with a 3x3 kernel size, has 64 filters, stride of 1 and padded to yield the same size as input, followed by a ReLU activation layer
#     
#     b. Max pooling layer of 2x2
#     
#     c. 2D convolution layer with a 3x3 kernel size, has 32 filters, stride of 1 and padded to yield the same size as input, followed by a ReLU activation layer
#     
#     d. Max pooling layer of 2x2
#     
#     e. Dense layer with 256 dimensions and ReLU as the activation layer

# In[17]:


model2 = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(kernel_size=(3,3), activation='relu', padding='same',
                                       strides=1, input_shape=(256,256,3), filters=128),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)),
        tf.keras.layers.Conv2D(kernel_size=(3,3), activation='relu', padding='same',
                                       strides=1, input_shape=(256,256,3), filters=32),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)),
        tf.keras.layers.Dense(256, activation="relu")
        tf.keras.layers.Dense(128, activation="relu"),
    ]
)
model2.compile(tf.keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")


# 6. Train the classifier again for 20 epochs with 100 steps per epoch. Also use the validation data during training the estimator.

# In[19]:


model.fit(INPUT, OUTPUT, epochs=20, steps_per_epoch=100)


# 7. Plot the accuracy and the loss over epochs for train & validation sets

# In[21]:


#code here

