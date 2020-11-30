import network_definitions
import numpy as np
from perform_reconstruction import perform_reconstruction
from tqdm import tqdm
import astropy.units as u
import keras
from sklearn.model_selection import train_test_split

__all__ = ["RNNtrainer"]

class RNNtrainer:

    def __init__(self, corsika_reader, verbose=True, network_type="CRNN"):

        self.verbose = verbose
        self.corsika_reader = corsika_reader
        self.signal_images, self.signal_header, self.signal_hillas, self.signal_reconstructed = \
            None, None, None, None
        self.background_images, self.background_header, self.background_hillas, self.background_reconstructed = \
            None, None, None, None
        self.network = None
        self.network_type = network_type

    def read_and_process(self, file_list, min_tels=2, intensity_cut=80, local_distance=3, **kwargs):

        images, header, hillas_parameters, reconstructed_parameters = None, None, None, None
        for file_name in tqdm(file_list):

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

            self.corsika_reader.images = images_loaded
            images_loaded = self.corsika_reader.scale_to_photoelectrons()

            # Perform Hillas intersection style reconstruction on all events
            geometry = self.corsika_reader.get_camera_geometry()
            reconstructed, hillas, selected = \
                perform_reconstruction(images_loaded, geometry,
                                       self.corsika_reader.telescope_x_positions*u.m,
                                       self.corsika_reader.telescope_y_positions*u.m,
                                       min_tels=min_tels, intensity_cut=intensity_cut, local_distance=local_distance)

            # Copy values into output arrays
            if images is None:
                images = images_loaded[selected]
                header = header_loaded[selected]
                hillas_parameters = hillas
                reconstructed_parameters = reconstructed
            else:
                images = np.concatenate((images, images_loaded[selected]), axis=0)
                header = np.concatenate((header, header_loaded[selected]), axis=0)
                hillas_parameters = np.concatenate((hillas_parameters, hillas), axis=0)
                reconstructed_parameters = np.concatenate((reconstructed_parameters, reconstructed), axis=0)

        return images, header, hillas_parameters, reconstructed_parameters

    def read_signal_and_background(self, signal_files, background_files,
                                   min_tels=2, intensity_cut=80, local_distance=3):

        print("Reading signal files...")
        self.signal_images, self.signal_header, self.signal_hillas, self.signal_reconstructed = \
            self.read_and_process(signal_files, min_tels=min_tels, intensity_cut=intensity_cut,
                                  local_distance=local_distance)

        print("Reading background files...")
        self.background_images, self.background_header, self.background_hillas, self.background_reconstructed = \
            self.read_and_process(background_files, min_tels=min_tels, intensity_cut=intensity_cut,
                                  local_distance=local_distance)

    @staticmethod
    def save_processed_images(output_file,  signal_images, signal_header, signal_hillas, signal_reconstructed,
                              background_images, background_header, background_hillas, background_reconstructed):

        np.savez_compressed(output_file, signal_images=signal_images,
                            signal_header=signal_header,signal_hillas=signal_hillas,
                            signal_reconstructed=signal_reconstructed,
                            background_images=background_images,
                            background_header=background_header, background_hillas=background_hillas,
                            background_reconstructed=background_reconstructed)
    @staticmethod
    def load_processed_images(input_file):
        loaded = np.load(input_file)

        return loaded["signal_images"].astype("float32"), loaded["signal_header"].astype("float32"), \
               loaded["signal_hillas"].astype("float32"), loaded["signal_reconstructed"].astype("float32"), \
               loaded["background_images"].astype("float32"), loaded["background_header"].astype("float32"), \
               loaded["background_hillas"].astype("float32"), loaded["background_reconstructed"].astype("float32")

    def save_training_images(self, output_file):

        self.save_processed_images(output_file, self.signal_images, self.signal_header,
                                   self.signal_hillas, self.signal_reconstructed,
                                   self.background_images, self.background_header,
                                   self.background_hillas, self.background_reconstructed)

    def load_training_images(self, input_file):

        self.signal_images, self.signal_header, \
        self.signal_hillas, self.signal_reconstructed,\
        self.background_images, self.background_header, \
        self.background_hillas, self.background_reconstructed = self.load_processed_images(input_file)

    def create_network(self, network_type):

        self.network_type = network_type

        if self.network_type is "CRNN":
            input_shape = (self.signal_images.shape[1], self.signal_images.shape[2], self.signal_images.shape[3], 1)
            input_layer, output_layer = network_definitions.create_recurrent_cnn(input_shape,
                                                                                 self.signal_hillas.shape[1:])

        if self.network_type is "HillasRNN":
            input_layer, output_layer = network_definitions.create_hillas_rnn(self.signal_hillas.shape[1:])

        sgd = keras.optimizers.Adam(lr=0.0005)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.network = model

    def train_network(self, output_file):

        hillas_input = np.concatenate((self.signal_hillas, self.background_hillas)).astype("float32")
        image_input = np.concatenate((self.signal_images, self.background_images)).astype("float32")

        image_input = image_input.reshape((image_input.shape[0], image_input.shape[1],
                                           image_input.shape[2], image_input.shape[3], 1))
        image_sum = np.sum(image_input.reshape((image_input.shape[0], image_input.shape[1],
                                                image_input.shape[2] * image_input.shape[3])), axis=-1, dtype="float32")
        image_input /= image_sum[..., np.newaxis, np.newaxis, np.newaxis]
        image_input[np.isnan(image_input)] = 0
        image_input[image_input < 0] = 0
        print(image_sum)

        image_mask = image_sum > 0
        image_mask = image_mask.astype("int")

        signal_target = np.zeros((2, self.signal_hillas.shape[0]))
        signal_target[0][:] = 1
        signal_target = signal_target.T

        background_target = np.zeros((2, self.background_hillas.shape[0]))
        background_target[1][:] = 1
        background_target = background_target.T
        target = np.concatenate((signal_target, background_target))

        stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0.0,
                                                 patience=20,
                                                 verbose=2, mode='auto')

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=10, min_lr=0.0)

        if self.network_type == "HillasRNN":
            hillas_input, val_hillas_input, target, val_target = train_test_split(hillas_input, target, test_size=0.2)
            fit = self.network.fit([hillas_input], target, epochs=1000,
                                   batch_size=100, validation_data=([val_hillas_input], val_target),
                                   shuffle=True,
                                   callbacks=[reduce_lr, stopping])

        elif self.network_type == "CRNN":
            image_input, val_image_input, image_mask, val_image_mask, \
            hillas_input, val_hillas_input, target, val_target = \
                train_test_split(image_input, image_mask, hillas_input, target, test_size=0.2)

            fit = self.network.fit([image_input, image_mask, hillas_input], target, epochs=1000,
                                   batch_size=100, validation_split=0.2, shuffle=True,
                                   callbacks=[reduce_lr, stopping])
