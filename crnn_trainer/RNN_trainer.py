import crnn_trainer.network_definitions
import numpy as np
from crnn_trainer.perform_reconstruction import perform_reconstruction
from tqdm import tqdm
import astropy.units as u
import keras
from sklearn.model_selection import train_test_split
from sparse import COO
import sparse
import pickle
from skimage.restoration import denoise_wavelet

__all__ = ["RNNtrainer"]


class RNNtrainer:

    def __init__(self, corsika_reader, verbose=True, network_type="CRNN"):

        self.verbose = verbose
        self.corsika_reader = corsika_reader
        self.signal_images, self.signal_header, \
            self.signal_hillas, self.signal_reconstructed = None, None, None, None

        self.background_images, self.background_header, \
            self.background_hillas, self.background_reconstructed = None, None, None, None
        self.network = None
        self.network_type = network_type

    def read_and_process(self, file_list, min_tels=2, intensity_cut=80, local_distance=3, denoise_sigma=0., **kwargs):

        images, header, hillas_parameters, reconstructed_parameters = None, None, None, None
        for file_name in tqdm(file_list):

            # Load up our images and header from the saved files
            loaded = np.load(file_name)
            images_loaded = loaded["images"].astype("float32")

            # Check to see if there are any images that are empty
            image_shape = images_loaded.shape
            image_sum = np.sum(images_loaded.reshape(image_shape[0],
                                                  image_shape[1]*image_shape[2]*image_shape[3]), axis=-1)
            not_empty = image_sum > 0
            images_loaded = images_loaded[not_empty]
            header_loaded = loaded["header"][not_empty]

            self.corsika_reader.images = images_loaded
            images_loaded = self.corsika_reader.scale_to_photoelectrons(**kwargs)

            # Perform Hillas intersection style reconstruction on all events
            geometry = self.corsika_reader.get_camera_geometry()
            reconstructed, hillas, selected = \
                crnn_trainer.perform_reconstruction(images_loaded, geometry,
                                                    self.corsika_reader.telescope_x_positions*u.m,
                                                    self.corsika_reader.telescope_y_positions*u.m,
                                                    min_tels=min_tels, intensity_cut=intensity_cut,
                                                    local_distance=local_distance)

            if denoise_sigma > 0.:

                    image_sum = np.max(images_loaded.reshape((images_loaded.shape[0], images_loaded.shape[1],
                                       images_loaded.shape[2] * images_loaded.shape[3])), axis=-1)
                    image_mask = image_sum != 0
                    image_mask = image_mask.astype("int")
                    
                    empty_mask = (images_loaded == 0)
                    image_input_shape = images_loaded.shape
                    image_mask_shape = image_mask.shape

                    images_loaded = images_loaded.reshape((image_input_shape[0] * image_input_shape[1],
                                                           image_input_shape[2], image_input_shape[3]))
                    image_mask = image_mask.reshape((image_mask_shape[0] * image_mask_shape[1]))

                    for dn in range(images_loaded.shape[0]):
                        if image_mask[dn]:
                            images_loaded[dn] = denoise_wavelet(images_loaded[dn], sigma=denoise_sigma)

                    images_loaded = images_loaded.reshape(image_input_shape)
                    image_mask = image_mask.reshape(image_mask_shape)
                    images_loaded[empty_mask] = 0

            # Copy values into output arrays
            if images is None:
                images = COO.from_numpy(images_loaded[selected].astype("float16"))
                header = header_loaded[selected]
                hillas_parameters = hillas
                reconstructed_parameters = reconstructed
            else:
                images = sparse.concatenate((images,  COO.from_numpy(images_loaded[selected].astype("float16"))), axis=0)
                header = np.concatenate((header, header_loaded[selected]), axis=0)
                hillas_parameters = np.concatenate((hillas_parameters, hillas), axis=0)
                reconstructed_parameters = np.concatenate((reconstructed_parameters, reconstructed), axis=0)

        return images, header, hillas_parameters, reconstructed_parameters

    # Read in and perform Hillas parameterisation and event reconstruction on our signal nad background files
    def read_signal_and_background(self, signal_files, background_files,
                                   min_tels=2, intensity_cut=80, local_distance=3, **kwargs):

        print("Reading signal files...")
        self.signal_images, self.signal_header, self.signal_hillas, self.signal_reconstructed = \
            self.read_and_process(signal_files, min_tels=min_tels, intensity_cut=intensity_cut,
                                  local_distance=local_distance, **kwargs)

        print("Reading background files...")
        self.background_images, self.background_header, self.background_hillas, self.background_reconstructed = \
            self.read_and_process(background_files, min_tels=min_tels, intensity_cut=intensity_cut,
                                  local_distance=local_distance, **kwargs)

    # Save processed images which can be used for input to network training
    @staticmethod
    def save_processed_images(output_file,  signal_images, signal_header, signal_hillas, signal_reconstructed,
                              background_images, background_header, background_hillas, background_reconstructed):

        data = [signal_images, signal_header, signal_hillas, signal_reconstructed,
                background_images, background_header, background_hillas, background_reconstructed]

        with open(output_file, "wb") as f:
            pickle.dump(data, f)
            f.close()

    # Load processed events to be used for training
    @staticmethod
    def load_processed_images(input_file):
        with open(input_file, "rb") as f:
            return pickle.load(f)

    def save_training_images(self, output_file):

        self.save_processed_images(output_file, self.signal_images, self.signal_header,
                                   self.signal_hillas, self.signal_reconstructed,
                                   self.background_images, self.background_header,
                                   self.background_hillas, self.background_reconstructed)

    def load_training_images(self, input_file):

        signal_images, signal_header, \
        signal_hillas, signal_reconstructed,\
        background_images, background_header, \
        background_hillas, background_reconstructed = self.load_processed_images(input_file)

        if self.signal_images is None:
            self.signal_images, self.signal_header, \
            self.signal_hillas, self.signal_reconstructed,\
            self.background_images, self.background_header, \
            self.background_hillas, self.background_reconstructed = signal_images, signal_header, \
                                                                    signal_hillas, signal_reconstructed,\
                                                                    background_images, background_header, \
                                                                    background_hillas, background_reconstructed
        else:
            self.signal_images = sparse.concatenate((self.signal_images, signal_images))
            self.signal_header = np.concatenate((self.signal_header, signal_header))
            self.signal_hillas = np.concatenate((self.signal_hillas, signal_hillas))
            self.signal_reconstructed = np.concatenate((self.signal_reconstructed, signal_reconstructed))

            self.background_images = sparse.concatenate((self.background_images, background_images))
            self.background_header = np.concatenate((self.background_header, background_header))
            self.background_hillas = np.concatenate((self.background_hillas, background_hillas))
            self.background_reconstructed = np.concatenate((self.background_reconstructed, background_reconstructed))

    # Create neural network of the requested type
    def create_network(self, network_type):

        self.network_type = network_type

        # Load in network of the type requested
        if self.network_type is "CRNN":
            input_shape = (self.signal_images.shape[1], self.signal_images.shape[2], self.signal_images.shape[3], 1)
            input_layer, output_layer = crnn_trainer.network_definitions.create_recurrent_cnn(input_shape,
                                                                                              self.signal_hillas.shape[1:],
                                                                                              hidden_nodes=32)
        elif self.network_type is "HillasRNN":
            input_layer, output_layer = crnn_trainer.network_definitions.create_hillas_rnn(self.signal_hillas.shape[1:])

        # Compile is using fairly standard parameters
        sgd = keras.optimizers.Adam(lr=0.001)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.network = model

    # Generator function to produce training inputs for Keras (done to save memory usage)
    def generate_training_image(self, batch_size=10000, particle_type="all",
                                dead_pixel_fraction=0.1, pixel_infill=True,
                                bg_weight=1.0, denoise_sigma=0.):

        signal_length = len(self.signal_hillas)

        # Produce a shuffled set of indices to let us choose s random sample
        if particle_type == "all":
            indices = np.arange(len(self.signal_hillas) + len(self.background_hillas))
            np.random.shuffle(indices)
        elif particle_type == "signal":
            indices = np.arange(len(self.signal_hillas))
        elif particle_type == "background":
            indices = np.arange(len(self.background_hillas))
            signal_length = 0

        # Then start our generator loop
        index = 0
        while True:
            # Grab our selection and decide whether they are signal or not
            selection = indices[index:index+batch_size]
            signal_selection = selection[selection < signal_length]
            background_selection = selection[selection > signal_length-1] - signal_length

            # Create the target arrays
            signal_target = np.zeros((2, len(signal_selection)))
            signal_target[0][:] = 1
            background_target = np.zeros((2, len(background_selection)))
            background_target[1][:] = 1
            signal_weight = np.ones(len(signal_selection))
            background_weight = np.ones(len(background_selection)) * bg_weight

            # Then create our training sets
            if particle_type == "all":
                hillas_input = np.concatenate((self.signal_hillas[signal_selection],
                                               self.background_hillas[background_selection]))
                target = np.concatenate((signal_target.T, background_target.T))
                weight = np.concatenate((signal_weight, background_weight))
            elif particle_type == "signal":
                hillas_input = self.signal_hillas[signal_selection]
                target = signal_target.T
                weight = signal_weight
            elif particle_type == "background":
                hillas_input = self.background_hillas[background_selection]
                target = background_target.T
                weight = background_weight

            if self.network_type == "HillasRNN":
                yield hillas_input, target, weight
            elif self.network_type == "CRNN":

                # For CRNN we need to use our image input
                if particle_type == "all":
                    image_input = np.concatenate((self.signal_images[signal_selection].todense().astype("float32"),
                                                  self.background_images[background_selection].todense().astype("float32")))
                elif particle_type == "signal":
                    image_input = self.signal_images[signal_selection].todense().astype("float32")
                elif particle_type == "background":
                    image_input = self.background_images[background_selection].todense().astype("float32")

                if dead_pixel_fraction > 0.:
                    dead_pix = np.random.rand(*image_input.shape[1:]) > dead_pixel_fraction
                    dead_pix = dead_pix[np.newaxis, :]
                    image_input *= dead_pix

                    if pixel_infill:
                        self.infill_image(image_input, dead_pix)

                image_input = image_input.reshape((image_input.shape[0], image_input.shape[1],
                                                   image_input.shape[2], image_input.shape[3], 1))
                # Normalise the images to peak at 1
                image_sum = np.max(image_input.reshape((image_input.shape[0], image_input.shape[1],
                                                        image_input.shape[2] * image_input.shape[3])), axis=-1)
                image_input /= image_sum[..., np.newaxis, np.newaxis, np.newaxis]
                image_input[np.isnan(image_input)] = 0
                image_input[image_input < 0] = 0

                # Finally create mask for empty images
                image_mask = image_sum != 0
                image_mask = image_mask.astype("int")

                # Perform image denoising
                if denoise_sigma > 0.:
                    empty_mask = image_input == 0
                    image_input_shape = image_input.shape
                    image_mask_shape = image_mask.shape

                    image_input = image_input.reshape((image_input_shape[0] * image_input_shape[1],
                                                       image_input_shape[2], image_input_shape[3]))
                    image_mask = image_mask.reshape((image_mask_shape[0] * image_mask_shape[1]))

                    for dn in range(image_input.shape[0]):
                        if image_mask[dn]:
                            image_input[dn] = denoise_wavelet(image_input[dn], sigma=denoise_sigma)

                    image_input = image_input.reshape(image_input_shape)
                    image_mask = image_mask.reshape(image_mask_shape)
                    image_input[empty_mask] = 0

                yield [image_input, image_mask, hillas_input], target, weight

            index += batch_size
            if index > len(indices):
                index = 0
                self.target = None

    # Method for infilling missing pixels with the average of its neighbours in the case
    # that some are deactivated
    @staticmethod
    def infill_image(image_input, dead_pix):
        # Expand our images out by one pixel in all directions
        expanded_shape = (image_input.shape[0], image_input.shape[1],
                          image_input.shape[2] + 2, image_input.shape[3] + 2)
        # Then shift our image one pixel in each direction
        shift_left = np.zeros(expanded_shape)
        shift_left[:, :, 0:image_input.shape[2], 1:image_input.shape[3] + 1] = image_input

        shift_right = np.zeros(expanded_shape)
        shift_right[:, :, 2:image_input.shape[2] + 2, 1:image_input.shape[3] + 1] = image_input

        shift_up = np.zeros(expanded_shape)
        shift_up[:, :, 1:image_input.shape[2] + 1, 0:image_input.shape[3]] = image_input

        shift_down = np.zeros(expanded_shape)
        shift_down[:, :, 1:image_input.shape[2] + 1, 2:image_input.shape[3] + 2] = image_input

        # Then sum these shifted images
        sum_shifts = shift_left + shift_right + shift_up + shift_down
        non_dead_pixels = (shift_left > 0).astype("int") + (shift_right > 0).astype("int") + \
                          (shift_up > 0).astype("int") + (shift_down > 0).astype("int")
        # And divide by number of pixels summed
        sum_shifts = sum_shifts / non_dead_pixels.astype("float32")

        # Finally fill in blanks in the case that pixels have no neighbours
        sum_shifts[np.isinf(sum_shifts)] = 0
        sum_shifts[np.isnan(sum_shifts)] = 0

        # Finally take the centre of our expanded image
        sum_shifts = sum_shifts[:, :, 1:image_input.shape[2] + 1, 1:image_input.shape[3] + 1]
        dead_pix = np.invert(np.repeat(dead_pix, image_input.shape[0], axis=0))
        # And fill in the deactivated pixels with our average values
        image_input[dead_pix] = sum_shifts[dead_pix]

    # Train our chosen network
    def train_network(self, output_file, batch_size=1000, validation_fraction=0.2, bg_weight=1.0, denoise_sigma=0.):

        print("Training", self.network_type, "network with", len(self.signal_hillas), "signal events and",
              len(self.background_hillas), "background events")

        stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0.0,
                                                 patience=20,
                                                 verbose=2, mode='auto')

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=10, min_lr=0.0)

        logger = keras.callbacks.CSVLogger("log_" + output_file + ".csv",
                                           separator=' ', append=False)

        checkpoint = keras.callbacks.ModelCheckpoint("check_" + output_file + ".h5",
                                                     monitor='val_loss', save_best_only=True)

        total_length = (len(self.signal_hillas) + len(self.background_hillas))
        steps = total_length / batch_size
        val_steps = int(np.floor((total_length * validation_fraction)/batch_size))

        fit = self.network.fit(self.generate_training_image(batch_size=batch_size,
                                                            bg_weight=bg_weight,
                                                            denoise_sigma=denoise_sigma),
                               steps_per_epoch=steps-val_steps,
                               validation_data=self.generate_training_image(batch_size=batch_size,
                                                                            bg_weight=bg_weight,
                                                                            denoise_sigma=denoise_sigma),
                               validation_steps=val_steps,
                               epochs=500, callbacks=[reduce_lr, stopping, logger, checkpoint], shuffle=True)

        self.network.save_weights(output_file)

    # Load pre-trained network weights
    def load_network(self, weight_file):
        self.network.load_weights(weight_file)

    # Run network on stored signal and BG data
    def test_signal_and_background(self):
        return self.test_network(self.signal_hillas.astype("float32"), self.signal_images.astype("float32")),  \
               self.test_network(self.background_hillas.astype("float32"), self.background_hillas.astype("float32"))

    # Evaluate network performance on a given dataset
    def test_network(self, batch_size=1000, dead_pixel_fraction=0., pixel_infill=False, denoise_sigma=0.):

        # Perform normalisation as in training
        total_length = len(self.signal_hillas)
        steps = total_length / batch_size
        signal_prediction = self.network.predict(self.generate_training_image(batch_size=batch_size,
                                                                              particle_type="signal",
                                                                              dead_pixel_fraction=dead_pixel_fraction,
                                                                              pixel_infill=pixel_infill,
                                                                              denoise_sigma=denoise_sigma),
                                                 steps=steps, verbose=1)

        total_length = len(self.background_hillas)
        steps = total_length / batch_size
        background_prediction = self.network.predict(self.generate_training_image(batch_size=batch_size,
                                                                                  particle_type="background",
                                                                                  dead_pixel_fraction=dead_pixel_fraction,
                                                                                  pixel_infill=pixel_infill,
                                                                                  denoise_sigma=denoise_sigma),
                                                     steps=steps, verbose=1)

        return signal_prediction.T[0], background_prediction.T[0]
