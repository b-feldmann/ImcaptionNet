from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input
import numpy as np

from imcaption_net import ImcaptionNet

if __name__ == '__main__':
    img_path = '.data/train2014/COCO_train2014_000000000009.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    net = ImcaptionNet()

    features = net.predict(x)
    print(features.shape)
