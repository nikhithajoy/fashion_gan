import logging
import tensorflow_datasets as tfds
import tensorflow as tf

logger = logging.getLogger(__name__)

class FashionMNISTLoader:
    def __init__(self, batch_size=128, buffer_size=60000):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def scale_images(self, data):
        image = data['image']
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def load(self):
        logger.info("Loading Fashion MNIST dataset...")
        ds = tfds.load('fashion_mnist', split='train', as_supervised=False)
        ds = ds.map(self.scale_images)
        ds = ds.cache()
        ds = ds.shuffle(self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        logger.info("Dataset loaded and preprocessed.")
        return ds
