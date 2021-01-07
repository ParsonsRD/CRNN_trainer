import os

from crnn_trainer.RNN_trainer import RNNtrainer
import numpy as np

trainer = RNNtrainer(None)
trainer.load_training_images("train_data_1.0.pickle")
trainer.create_network("CRNN")
trainer.train_network("test_net", 100)