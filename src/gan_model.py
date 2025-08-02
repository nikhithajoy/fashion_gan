import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class FashionGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim=128):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        logger.info("FashionGAN compiled with optimizers and loss functions.")

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Generate fake images
        generated_images = self.generator(random_latent_vectors, training=True)

        # Combine real and fake images
        combined_images = tf.concat([real_images, generated_images], axis=0)

        # Labels for real (0) and fake (1) images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        # Add noise to labels - improves GAN training stability
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.d_loss_fn(labels, predictions)

        grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train generator
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            predictions = self.discriminator(generated_images, training=False)
            g_loss = self.g_loss_fn(misleading_labels, predictions)

        grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        logger.debug(f"Discriminator loss: {d_loss.numpy()}, Generator loss: {g_loss.numpy()}")

        return {"d_loss": d_loss, "g_loss": g_loss}
