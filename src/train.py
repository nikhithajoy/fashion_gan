import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from src.data_loader import FashionMNISTLoader
from src.generator import Generator
from src.discriminator import Discriminator
from src.gan_model import FashionGAN
from src.callbacks import ModelMonitor

logger = logging.getLogger(__name__)

def run_training(epochs=20):
    logger.info("Starting GAN training...")

    # Load dataset
    data_loader = FashionMNISTLoader()
    dataset = data_loader.load()

    # Build models
    gen = Generator()
    disc = Discriminator()

    generator_model = gen.get_model()
    discriminator_model = disc.get_model()

    # Initialize GAN model
    gan = FashionGAN(generator_model, discriminator_model)

    # Compile GAN model
    gan.compile(
        g_optimizer=Adam(learning_rate=0.0001),
        d_optimizer=Adam(learning_rate=0.00001),
        g_loss_fn=BinaryCrossentropy(),
        d_loss_fn=BinaryCrossentropy()
    )

    # Training with callbacks
    history = gan.fit(dataset, epochs=epochs, callbacks=[ModelMonitor()])
    logger.info("Training completed.")
    return history
