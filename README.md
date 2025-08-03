# Fashion GAN Demo

A clean and modular implementation of a Generative Adversarial Network (GAN) for generating images based on the Fashion MNIST dataset. This project demonstrates a scalable architecture using OOP principles, PEP8 compliance, logging, callbacks, and model monitoring — making it suitable for both experimentation and production-grade pipelines.

---

## Implementation Details
- Dataset: Fashion MNIST via tensorflow_datasets
- Architecture: Deep Convolutional GAN with 4 upsampling/downsampling blocks
- Framework: TensorFlow 2.x (Keras API)
- Loss: Binary Cross-Entropy
- Optimizers: Adam for both generator and discriminator

---

## Configuration
The default settings for batch size, latent dimension, and number of epochs are defined in the respective modules:

- batch_size=128 
- latent_dim=128
- epochs=20

You can modify them directly in src/train.py or via CLI enhancements.