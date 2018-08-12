import tensorflow as tf
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

# Normalize the dataset
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)


## Loss and Acccuracy
val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss,val_acc)

# predict

pred=model.predict([x_test])
import numpy as np
print(np.argmax(pred[0]))

# Seee the plot/image
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

## To save a model & to use it.

model.save('My_model_for_mnist')
new_model=tf.keras.models.load_model('My_model_for_mnist')

## use it
# predict

pred=new_model.predict([x_test])
import numpy as np
print(np.argmax(pred[26]))


plt.imshow(x_test[26],cmap=plt.cm.binary)
plt.show()

