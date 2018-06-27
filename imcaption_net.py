import tensorflow as tf
from tensorflow.python.keras.models import Model


class ImcaptionNet(Model):
    """docstring for ImcaptionNet"""

    def __init__(self):
        super().__init__(name='imcaption_net')
        base_resnet50 = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=None, pooling=None)
        # EXPERIMENT: Testen ob wir die Activation (-2) oder das Pooling (-1) benutzen
        self.input = base_resnet50.input
        self.encoder = Model(
            inputs=self.input, output=base_resnet50.layers[-2].output)

    def build(self, input_shape):
        # TODO Decoder (LSTM)
        super().build(input_shape)

    def call(self, inputs):
        return self.encoder(inputs)

    def compute_output_shape(self, input_shape):
        pass
