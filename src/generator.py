from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2D, LeakyReLU, UpSampling2D
import logging

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, input_dim=128):
        self.input_dim = input_dim
        self.model = self._build()

    def _build(self):
        logger.info("Building generator model...")
        model = Sequential(name="Generator")

        model.add(Dense(7 * 7 * 128, input_dim=self.input_dim))
        model.add(LeakyReLU(0.2))
        model.add(Reshape((7, 7, 128)))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=5, padding='same'))
        model.add(LeakyReLU(0.2))

        model.add(UpSampling2D())
        model.add(Conv2D(1, kernel_size=5, padding='same'))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(128, kernel_size=4, padding='same'))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(128, kernel_size=4, padding='same'))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(1, kernel_size=4, padding='same', activation='sigmoid'))

        logger.info("Generator model built successfully.")
        return model

    def get_model(self):
        return self.model
