import os

from tqdm import tqdm
import numpy as np
from crnn_trainer.network_definitions import create_denoising_autoencoder
from corsika_toy_iact.iact_array import IACTArray

directory = "/Users/dparsons/lustre/fs20/group/hess/user/dparsons/HadronicInteractions/IACT_Spec/QGSII/corsika/Gamma/alt1800/Zenith0/"
filelist = os.listdir(directory)
signal_file = []

for file_name in filelist:
    if "image.npz" in file_name:
        signal_file.append(directory+file_name)

directory = "/Users/dparsons/lustre/fs20/group/hess/user/dparsons/HadronicInteractions/IACT_Spec/QGSII/corsika/Proton/alt1800/Zenith0/"
filelist = os.listdir(directory)
background_file = []

for file_name in filelist:
    if "image.npz" in file_name:
        background_file.append(directory+file_name)

x = np.linspace(-120, 120, 3)
xx, yy = np.array(np.meshgrid(x, x))
positions = np.vstack((xx.ravel(), yy.ravel())).T
corsika_reader = IACTArray(positions, radius=6)

images_noisy, images_no_noise = None, None

for file_name in tqdm(signal_file[0:2]):

    # Load up our images and header from the saved files
    loaded = np.load(file_name)
    images_loaded = loaded["images"]

    # Check to see if there are any images that are empty
    image_shape = images_loaded.shape
    image_sum = np.sum(images_loaded.reshape(image_shape[0],
                                             image_shape[1]*image_shape[2]*image_shape[3]), axis=-1)
    not_empty = image_sum > 0
    images_loaded = images_loaded[not_empty]
    header_loaded = loaded["header"][not_empty]


    corsika_reader.images = images_loaded

    if images_noisy is None:
        images_no_noise = corsika_reader.scale_to_photoelectrons(pedestal_width=0)
        images_noisy = corsika_reader.scale_to_photoelectrons(pedestal_width=1)
    else:
        images_no_noise = np.concatenate((images_no_noise, corsika_reader.scale_to_photoelectrons(pedestal_width=0)))
        images_noisy = np.concatenate((images_noisy, corsika_reader.scale_to_photoelectrons(pedestal_width=1)))

image_shape = images_no_noise.shape
images_noisy = images_noisy.reshape(image_shape[0]*image_shape[1], image_shape[2], image_shape[3], 1)
images_no_noise = images_no_noise.reshape(image_shape[0]*image_shape[1], image_shape[2], image_shape[3], 1)

image_selection = np.sum(images_no_noise.reshape(images_no_noise.shape[0],
                                                 images_no_noise.shape[1]*images_no_noise.shape[2]), axis=-1) > 10

images_noisy_sum = np.sum(images_noisy.reshape(images_noisy.shape[0],
                                               images_noisy.shape[1]*images_noisy.shape[2]), axis=-1)

images_noisy = images_noisy / images_noisy_sum[..., np.newaxis, np.newaxis, np.newaxis]
images_noisy[images_noisy < 0] = 0

images_no_noise = images_no_noise / images_noisy_sum[..., np.newaxis, np.newaxis, np.newaxis]
images_no_noise[images_no_noise < 0] = 0
images_no_noise[images_no_noise > 1] = 1

print(images_noisy.shape, images_noisy[image_selection].shape)
import keras
input_layer, output_layer = create_denoising_autoencoder(images_noisy.shape[1:])
sgd = keras.optimizers.Adam(lr=0.0005)
model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(images_noisy[image_selection], images_no_noise[image_selection], epochs=10,
                                   batch_size=1000,
                                   validation_split=0.2,
                                   shuffle=True)
