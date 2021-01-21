import os

from crnn_trainer.RNN_trainer import RNNtrainer
import numpy as np
import matplotlib.pyplot as plt

ped = "0.0"
dead_pixel_fraction = [0.]
dir = "./data/"
trainer = RNNtrainer(None)

signal, bg = [], []
header = "Event SimulatedEnergy SimulatedAzimuth SimulatedZenith X0 SimulatedCoreX SimulatedCoreY " \
         "ReconstructedNomX ReconstructedNomY ReconstructedCoreX ReconstructedCoreY Eta"

for pix_frac in dead_pixel_fraction:
    print("Loading File")
    trainer.load_training_images(dir + "test_data_" + ped + "_0.04")
    trainer.create_network("CRNN")
    print("Loading Network")
    trainer.load_network(dir + "network/check_network_1.0_0.04.h5")
    print("Analysing Events")
    signal_prediction, background_prediction = trainer.test_network(dead_pixel_fraction=pix_frac)
    signal.append(-1 * np.log10(1-signal_prediction))
    bg.append(-1 * np.log10(1-background_prediction))

    signal_prediction = signal_prediction.reshape((signal_prediction.shape[0], 1))
    signal_prediction = -1 * np.log10(1 - signal_prediction)
    background_prediction = background_prediction.reshape((background_prediction.shape[0], 1))
    background_prediction = -1 * np.log10(1 - background_prediction)

    signal_output = np.concatenate((trainer.signal_header.T, trainer.signal_reconstructed.T, signal_prediction.T)).T
    bg_output = np.concatenate((trainer.background_header.T, trainer.background_reconstructed.T, background_prediction.T)).T
    np.savetxt("output/signal_"+ped+"_0.04_" + str(pix_frac) + ".out", signal_output, header=header)
    np.savetxt("output/background_"+ped+"_0.04_" + str(pix_frac) + ".out", bg_output, header=header)

    print(signal_output.shape, bg_output.shape)

cut = np.percentile(signal[0], 20)
print(cut)

for i in range(len(signal)):
    print(np.sum(signal[i] > cut)/len(signal[i]))
    print(np.sum(bg[i] > cut)/len(bg[i]))