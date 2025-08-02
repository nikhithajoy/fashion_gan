from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense
import logging

logger = logging.getLogger(__name__)

class Discriminator:
    def __init__(self, input_shape=(28, 28, 1)):
        self.input_shape = input_shape
        self.model = self._build()

    def _build(self):
        logger.info("Building discriminator model...")
        model = Sequential(name="Discriminator")

        model.add(Conv2D(32, kernel_size=5, input_shape=self.input_shape))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(64, kernel_size=5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, kernel_size=5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, kernel_size=5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        logger.info("Discriminator model built successfully.")
        return model

    def get_model(self):
        return self.model
