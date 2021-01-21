import keras
from keras import layers

__all__ = ["create_hillas_rnn", "create_recurrent_cnn", "create_denoising_autoencoder"]


def create_hillas_rnn(input_shape=(9, 6), hidden_nodes=64):
    input_layer = keras.Input(shape=input_shape)
    lstm_1 = layers.Bidirectional(layers.LSTM(hidden_nodes, activation="relu",
                                              return_sequences=False))(input_layer)
    dense_1 = layers.Dense(hidden_nodes, activation="relu")(lstm_1)
    drop_1 = layers.Dropout(rate=0.5)(dense_1)
    dense_2 = layers.Dense(hidden_nodes, activation="relu")(drop_1)

    output_layer = layers.Dense(2, activation="softmax")(dense_2)

    return input_layer, output_layer


def create_recurrent_cnn(cnn_input_shape=(9, 40, 40, 1), hillas_input_shape=(9, 6),
                          hidden_nodes=64, filters=20):
    # OK first we have out CNN input layer, it's time distributed to account for the multiple telescope types
    print(cnn_input_shape)
    input_layer = keras.Input(shape=cnn_input_shape)
    conv_lstm_1 = layers.TimeDistributed(layers.Conv2D(filters=filters, activation="relu",
                                                       kernel_size=(3, 3)))(input_layer)
    max_pooling_1 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(conv_lstm_1)
    conv_lstm_2 = layers.TimeDistributed(layers.Conv2D(filters=filters, activation="relu",
                                                       kernel_size=(3, 3)))(max_pooling_1)
    max_pooling_2 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(conv_lstm_2)

    # Once our CNN is done with we can flatten it out  and perform some dropout
    flatten = layers.TimeDistributed(layers.Flatten())(max_pooling_2)
    drop_1 = layers.TimeDistributed(layers.Dropout(rate=0.5))(flatten)
    dense_1 = layers.TimeDistributed(layers.Dense(hidden_nodes, activation="relu"))(drop_1)
    drop_2 = layers.TimeDistributed(layers.Dropout(rate=0.5))(dense_1)

    # Here we have to do something a bit tricky, introduce a masking input to allow us to ignore
    # telescopes that don't have an image
    input_mask = layers.Input(shape=(cnn_input_shape[0], 1))

    # Multiply our CNN by this to make sure it is ignored when we have no input
    mult_1 = layers.multiply([input_mask, drop_2])
    # We can then mask and run our RNN
    mask_mult1 = layers.Masking(mask_value=0)(mult_1)
    dense_c2 = layers.LSTM(hidden_nodes, return_sequences=True, activation='relu')(mask_mult1)
#    dense_c2 = layers.Bidirectional(layers.LSTM(hidden_nodes, activation='relu',
#                                                unroll=True, return_sequences=True))(mask_mult1)

    # Then we have our Hillas input layer
    input_hillas = layers.Input(shape=hillas_input_shape)
    mask2 = layers.Masking(mask_value=0)(input_hillas)
    dense_hillas_1 = layers.LSTM(hidden_nodes, return_sequences=True, activation='relu')(mask2)
    #layers.Bidirectional(layers.LSTM(hidden_nodes, return_sequences=True,
                      #                                activation="relu"))(mask2)
    drop_4 = layers.TimeDistributed(layers.Dropout(rate=0.5))(dense_hillas_1)

    # Combine our CNN and hillas layers
    merge = layers.concatenate([dense_c2, drop_4])

    # Make sure our masking is properly done
    mult = layers.multiply([input_mask, merge])
    mask_mult = layers.Masking(mask_value=0)(mult)

    # Finally feed all of this information into a final RNN
    lstm_1 = layers.LSTM(hidden_nodes, return_sequences=False, activation='relu')(mask_mult)#layers.Bidirectional(layers.LSTM(hidden_nodes,
                                              #activation='relu', return_sequences=False))(mask_mult)
    drop_3 = layers.Dropout(rate=0.5)(lstm_1)
    output = layers.Dense(2, activation='softmax')(drop_3)

    return [input_layer, input_mask, input_hillas], output


def create_denoising_autoencoder(input_shape=(40, 40, 1), hidden_nodes=64, filters=20):
    # Create the model
    input_layer = keras.Input(shape=input_shape)
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Conv2D(filters=filters/2, kernel_size=(3, 3), activation='relu', padding='same')(x)
    #x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    #x = layers.Conv2D(filters=filters/4, kernel_size=(3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # decoding
    x = layers.Conv2D(filters=filters/4, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(size=(2, 2))(x)
    #x = layers.Conv2D(filters=filters/2, kernel_size=(3, 3), activation='relu', padding='same')(x)
    #x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(x)  # <= padding='valid'!
    x = layers.UpSampling2D(size=(2, 2))(x)
    decoded = layers.Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same')(x)

    return input_layer, decoded
