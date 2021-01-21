import os

from crnn_trainer.RNN_trainer import RNNtrainer
import numpy as np
dir = "./data/"
ped = "1.0"
trainer = RNNtrainer(None)
trainer.load_training_images(dir + "test_data_" + ped + "_0.04")
trainer.create_network("CRNN")
trainer.train_network("test_net", 100)