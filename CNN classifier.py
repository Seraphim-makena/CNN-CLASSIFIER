import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models,datasets

data=datasets.mnist.load_data()


(x_train,Y_train),(x_test,Y_test)= data

fig=plt.figure(figsize=(5,5))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.imshow(x_train[i],cmap=plt.cm.binary)
plt.show()

from tensorflow.keras.utils import to_categorical
Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

print("Original data shape")
print(x_train.shape,Y_train.shape)
print(x_test.shape,Y_test.shape)

print("Reshaped data shape")
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

print(x_train.shape,Y_train.shape)
print(x_test.shape,Y_test.shape)


x_train=x_train/255.0
x_test=x_test/255.0



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,Y_train,epochs=10,validation_split=0.2,)

print("Evaluating the model...")
model.evaluate(x_test, Y_test)

print("Making predictions...")
y_prediction = model.predict(x_test)
print(y_prediction)


image_to_predict = x_test[0]
plt.imshow(image_to_predict.reshape(28, 28), cmap='gray')
plt.show()

predicted_number=model.predict(image_to_predict.reshape(1, 28, 28, 1))
print("The number is:",predicted_number.argmax())