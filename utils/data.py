# utils/data_loader.py

import numpy as np
from tensorflow.keras.datasets import mnist

class MnistDigits:
    def __init__(self, original, noised):
        """
        :param original: A numpy array of original MNIST images.
        :param noised: A numpy array of noised MNIST images.
        :return: An instance of MnistDigits with original and noised attributes.
        """
        self.original = original
        self.noised = noised

def load_mnist_data(size=3, error_rate=0.14) -> MnistDigits:
    """
    Fetches a subset of MNIST dataset, preprocesses the images and returns them along with their noised versions.

    :param size: The number of images to select from the MNIST dataset. Defaults to 3.
    :param error_rate: The probability of flipping a bit in a noised image. Defaults to 0.14.
    :return: An instance of MnistDigits with original and noised attributes.
    """
    
    (train_images, train_labels), _ = mnist.load_data()
    # Step 1: Collect one instance of each digit (0-9)
    digit_images = []
    for digit in range(10):
        # Find the first occurrence of each digit
        digit_index = np.where(train_labels == digit)[0][0]
        digit_images.append(train_images[digit_index])
    print(f'Collected {len(digit_images)} each digit')
    # Step 2: Convert list to an array and slice according to the specified size
    selected_indices = np.random.choice(len(digit_images), size=size, replace=False)
    # selected_indices = [1,4,9]
    selected_images = np.array([digit_images[i] for i in selected_indices])
    
    # Preprocess selected images
    train_images = preprocess_images(selected_images)
    
    # Nothing below this line should be changed
    noised_images = add_noise(train_images, error_rate)
    return MnistDigits(original=train_images, noised=noised_images)

def preprocess_images(images):
    images = images.reshape(-1, 28 * 28)
    images = (images > 127).astype(int)
    images = 2 * images - 1
    return images

def add_noise(images, error_rate=0.04):
    noisy_images = images.copy()
    for image in noisy_images:
        n_flip = int(error_rate * len(image))
        flip_indices = np.random.choice(len(image), n_flip, replace=False)
        image[flip_indices] *= -1
    return noisy_images
