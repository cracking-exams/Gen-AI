# --- STEP 1: Setup ---
!pip install tensorflow tensorflow_datasets matplotlib tqdm -q

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- STEP 2: Load MNIST dataset ---
(ds_train, _), ds_info = tfds.load(
    'mnist', split=['train', 'test'], as_supervised=True, with_info=True
)

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = (x - 0.5) * 2.0  # normalize to [-1, 1]
    return x, y

train_ds = ds_train.map(preprocess).batch(128)

# --- STEP 3: Define a small UNet-like model for DDPM inference ---
def get_unet():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(1, 3, padding="same")(x)
    return tf.keras.Model(inputs, x, name="unet")

model = get_unet()

# --- STEP 4: Load or fake pre-trained weights (for demo) ---
# Normally, you'd load a trained checkpoint, but we skip training for brevity.
# model.load_weights("pretrained_ddpm_mnist.h5")  # Uncomment if you have one

# --- STEP 5: Define diffusion sampling process (simplified) ---
T = 100  # number of timesteps
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

def sample(model, n=9):
    x = tf.random.normal((n, 28, 28, 1))  # start from noise
    for t in tqdm(reversed(range(T)), desc="Diffusion sampling"):
        alpha_t = alphas[t]
        alpha_cum_t = alphas_cumprod[t]
        z = tf.random.normal(x.shape) if t > 0 else 0
        pred_noise = model(x, training=False)
        x = (1 / np.sqrt(alpha_t)) * (x - (1 - alpha_t) / np.sqrt(1 - alpha_cum_t) * pred_noise) + np.sqrt(betas[t]) * z
    return x

# --- STEP 6: Generate samples ---
samples = sample(model, n=4)
samples = (samples + 1) / 2.0  # rescale to [0,1]

# --- STEP 7: Display generated images ---
plt.figure(figsize=(4,4))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(samples[i, :, :, 0], cmap="gray")
    plt.axis("off")
plt.suptitle("Generated MNIST-like images")
plt.show()
