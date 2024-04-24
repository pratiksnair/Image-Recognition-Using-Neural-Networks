import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import streamlit as st
from PIL import Image

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:2000]
training_labels = training_labels[:2000]
testing_images = testing_images[:400]
testing_labels = testing_labels[:400]

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"loss:{loss}")
# print(f"Accuracy:{accuracy}")

# model.save('image_classifier.model')

# model = models.load_model('image_classifier.model')

# img = cv.imread("c3.jpg")
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img = cv.resize(img, (32, 32))

# plt.imshow(img, cmap=plt.cm.binary)

# prediction = model.predict(np.array([img])/255)
# index = np.argmax(prediction)
# print(f'Prediction is{class_names[index]}')

def main():
    st.title('hey man')
    st.write('Upload an image')
    file = st.file_uploader('Please upload an image', type=['jpg','png'])

    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        resized_image = image.resize((32,32))
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape((1,32,32,3))

        model = tf.keras.models.load_model('image_classifier.model')

        predictions = model.predict(img_array)
        class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

        fig, ax = plt.subplots()
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, predictions[0], align = 'center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title('Preds')

        st.pyplot(fig) 
    else:
        st.text('no upload')

if __name__ == '__main__':
    main()

