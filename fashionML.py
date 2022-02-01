import tensorflow as tf #TensorFlow and Keras
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow version:', tf.__version__)

mnistFashionDataset = tf.keras.datasets.fashion_mnist

(trainingImages, trainingLabels), (testImages, testLabels) = mnistFashionDataset.load_data()

classNames = ['T-shirt/top',
              'Trouser', 
              'Pullover', 
              'Dress', 
              'Coat', 
              'Sandal', 
              'Shirt', 
              'Sneaker', 
              'Bag', 
              'Ankle boot']

print('Names of classes in dataset:')
for className in classNames:
  print(className)

print('Shape of training images in dataset:', trainingImages.shape)
print('Number of training images:', trainingImages.shape[0])
print('Format of training images:', trainingImages.shape[1], 'x', trainingImages.shape[2], 'px')

print('Number of training labels', len(trainingLabels))

plt.figure()
plt.imshow(trainingImages[100])
plt.colorbar()
plt.grid(True)
plt.show()

trainingImages = trainingImages / 255.0
testImages = testImages / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(True)
  plt.imshow(trainingImages[i], cmap=plt.cm.binary)
  plt.xlabel(classNames[trainingLabels[i]])
plt.show()

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

model.fit(trainingImages, trainingLabels, epochs=10)
testLoss, testAccuracy = model.evaluate(testImages, testLabels, verbose=2)
print('Test accuracy:', testAccuracy)
