import os
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img

logger = logging.getLogger(__name__)

class ModelMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_images=3, latent_dim=128, output_dir='images'):
        super().__init__()
        self.num_images = num_images
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal((self.num_images, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors, training=False)
        generated_images = generated_images * 255
        generated_images = tf.clip_by_value(generated_images, 0, 255)
        generated_images = tf.cast(generated_images, tf.uint8)

        for i in range(self.num_images):
            img = array_to_img(generated_images[i])
            file_path = os.path.join(self.output_dir, f'generated_img_epoch{epoch + 1}_{i}.png')
            img.save(file_path)
            logger.info(f"Saved generated image to {file_path}")
