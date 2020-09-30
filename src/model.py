import os
import numpy as np
from keras.models import Model,load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image


IMAGE_SIZE = (224, 224)


class FeatureModel:
    """wrap keras model with feature space and kernel mapping"""
    @classmethod
    def from_h5(cls, filename_h5):
        os.path.exists(filename_h5)
        return cls(load_model(filename_h5))

    def __init__(self, keras_model: Model):
       self.model: Model = keras_model

    def arr_to_features(self, x: np.array) -> np.array:
        """map (224, 224, 3) image array to feature space"""
        model = self.model
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.layers[-2].output)
        intermediate_output = intermediate_layer_model.predict(x)
        return intermediate_output

    def image_to_features(self, fpath: Image) -> np.array:
        """map image named in fpath to feature space"""
        img = load_img(fpath, target_size=IMAGE_SIZE)
        img = img_to_array(img)
        img = img.reshape(1, *IMAGE_SIZE, 3)
        img = img.astype('float32')
        img = img - [123.68, 116.779, 103.939]
        return self.arr_to_features(img)

    def kernel(self, img1, img2):
        """evaluate kernel inferred from model"""
        if isinstance(img1, str) and isinstance(img2, str):
            return np.dot(self.image_to_features(img1), self.image_to_features(img2))
        else:
            return np.dot(self.arr_to_features(img1), self.arr_to_features(img2))


if __name__ == '__main__':
    mod = FeatureModel.from_h5("./final_model.h5")
    print(mod.image_to_features("./finalize_dogs_vs_cats/dogs/dog.0.jpg"))