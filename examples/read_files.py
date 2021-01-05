import os

from crnn_trainer.RNN_trainer import RNNtrainer
from corsika_toy_iact.iact_array import IACTArray
import numpy as np

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
iact_array = IACTArray(positions, radius=6)

pedestals = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
for ped in pedestals:
    trainer = RNNtrainer(iact_array, network_type="CRNN")
    trainer.read_signal_and_background(signal_file[0:50], background_file[0:100], pedestal_width=ped)
    trainer.save_training_images("train_data_"+str(ped))

    trainer = RNNtrainer(iact_array, network_type="CRNN")
    trainer.read_signal_and_background(signal_file[50:100], background_file[100:], pedestal_width=ped)
    trainer.save_training_images("test_data_"+str(ped))
