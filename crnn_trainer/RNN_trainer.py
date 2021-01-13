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

    def read_and_process(self, file_list, min_tels=2, intensity_cut=80, local_distance=3, **kwargs):

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
                                                    min_tels=min_tels, intensity_cut=intensity_cut, local_distance=local_distance)

            # Copy values into output arrays
            if images is None:
                images = COO.from_numpy(images_loaded[selected])
                header = header_loaded[selected]
                hillas_parameters = hillas
                reconstructed_parameters = reconstructed
            else:
                images = sparse.concatenate((images,  COO.from_numpy(images_loaded[selected])), axis=0)
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

        print(signal_images.shape)
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

        self.signal_images, self.signal_header, \
        self.signal_hillas, self.signal_reconstructed,\
        self.background_images, self.background_header, \
        self.background_hillas, self.background_reconstructed = self.load_processed_images(input_file)

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
        sgd = keras.optimizers.Adam(lr=0.0005)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.network = model

    # Generator function to produce training inputs for Keras (done to save memory usage)
    def generate_training_image(self, batch_size=10000):

        signal_length = len(self.signal_hillas)
        indices = np.arange(len(self.signal_hillas) + len(self.background_hillas))

        # Produce a shuffled set of indices to let us choose s random sample
        np.random.shuffle(indices)

        # Then start our generator loop
        index = 0
        while True:
            # Grab our selection and decide whether they are signal or not
            selection = indices[index:index+batch_size]
            signal_selection = selection[selection < signal_length]
            background_selection = selection[selection > signal_length] - signal_length

            # Create the target arrays
            signal_target = np.zeros((2, len(signal_selection)))
            signal_target[0][:] = 1
            background_target = np.zeros((2, len(background_selection)))
            background_target[1][:] = 1

            # Then create our training sets
            hillas_input = np.concatenate((self.signal_hillas[signal_selection],
                                           self.background_hillas[background_selection]))
            target = np.concatenate((signal_target.T, background_target.T))

            if self.network_type == "HillasRNN":
                yield hillas_input, target
            elif self.network_type == "CRNN":
                # For CRNN we need to use our image input
                image_input = np.concatenate((self.signal_images[signal_selection].todense(),
                                              self.background_images[background_selection].todense()))

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

                yield [image_input, image_mask, hillas_input], target

            index += batch_size
            if index > len(indices):
                index = 0

    # Train our chosen network
    def train_network(self, output_file, batch_size=1000, validation_fraction=0.2):

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
        val_steps = int(np.floor(total_length * validation_fraction))

        fit = self.network.fit(self.generate_training_image(batch_size=batch_size),
                               steps_per_epoch=steps-val_steps,
                               validation_data=self.generate_training_image(batch_size=batch_size),
                               validation_steps=val_steps,
                               epochs=100, callbacks=[reduce_lr, stopping, logger, checkpoint], shuffle=True)

        self.network.save_weights(output_file)

    # Load pre-trained network weights
    def load_network(self, weight_file):
        self.network.load_weights(weight_file)

    # Run network on stored signal and BG data
    def test_signal_and_background(self):
        return self.test_network(self.signal_hillas.astype("float32"), self.signal_images.astype("float32")),  \
               self.test_network(self.background_hillas.astype("float32"), self.background_hillas.astype("float32"))

    # Evaluate network performance on a given dataset
    def test_network(self, image_input, hillas_input):

        # Perform normalisation as in training
        image_input = image_input.reshape((image_input.shape[0], image_input.shape[1],
                                           image_input.shape[2], image_input.shape[3], 1))
        image_sum = np.max(image_input.reshape((image_input.shape[0], image_input.shape[1],
                                                image_input.shape[2] * image_input.shape[3])), axis=-1, dtype="float32")
        image_input /= image_sum[..., np.newaxis, np.newaxis, np.newaxis]
        image_input[np.isnan(image_input)] = 0
        image_input[image_input < 0] = 0

        image_mask = image_sum > 0
        image_mask = image_mask.astype("int")

        prediction = None
        if self.network_type == "HillasRNN":
            prediction = self.network.predict([hillas_input])
        elif self.network_type == "CRNN":
            prediction = self.network.predict([image_input, image_mask, hillas_input])

        return prediction
