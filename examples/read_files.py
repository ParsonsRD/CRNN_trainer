
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from RNN_trainer import RNNtrainer
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

trainer = RNNtrainer(iact_array, network_type="CRNN")
print(len(signal_file), len(background_file))
#trainer.read_signal_and_background(signal_file[0:100], background_file[0:100])
trainer.load_training_images("test_data.npz")
#trainer.save_training_images("train_data")
trainer.create_network("CRNN")
trainer.train_network("out")