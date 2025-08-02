import logging
import os
from src.train import run_training

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/fashion_gan.log',
        filemode='a',
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main():
    setup_logging()
    logging.info("Fashion GAN demo started.")
    run_training(epochs=20)
    logging.info("Fashion GAN demo finished.")

if __name__ == "__main__":
    main()
