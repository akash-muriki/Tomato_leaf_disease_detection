import cv2
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential, layers, preprocessing
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import glob


def con(i):
    k = i + "/*.JPG"
    l = []
    for i in glob.glob(k):
        img = preprocessing.image.load_img(i)
        l.append(img.__array__())
    return l


flowers_images_dict = {
    'bacterial_spot': con("archive/Tomato/Tomato-Leaf-Disease-Research-Dataset-Train-Valid/Tomato___Bacterial_spot"),
    'early_blight': con("archive/Tomato/Tomato-Leaf-Disease-Research-Dataset-Train-Valid/Tomato___Early_blight"),
    'healthy': con("archive/Tomato/Tomato-Leaf-Disease-Research-Dataset-Train-Valid/Tomato___healthy"),
    'light_blight': con("archive/Tomato/Tomato-Leaf-Disease-Research-Dataset-Train-Valid/Tomato___Late_blight"),
    'mold': con("archive/Tomato/Tomato-Leaf-Disease-Research-Dataset-Train-Valid/Tomato___Leaf_Mold"),
}
flowers_labels_dict = {
    'bacterial_spot': 0,
    'early_blight': 1,
    'healthy': 2,
    'light_blight': 3,
    'mold': 4,

}

X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        resized_img = cv2.resize(image, (224, 224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, trainable=False)

model = Sequential([
    pretrained_model_without_top_layer,
    layers.Dense(5, activation="sigmoid")
])

model.compile(optimizer='sgd',
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=3)

print(model.evaluate(X_test_scaled, y_test))

pred = model.predict(X_test_scaled)

print(classification_report(pred, y_test))
