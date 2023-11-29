from data_conversion import DataConversion
from model import WaveNet
import torch
from torch.utils.data import random_split

test1 = DataConversion('./dataset/*.mp3')
test1.load_data()
spect_list_mel = test1.data_to_mel()
test1.display_mel(spect_list_mel, 20)

trainSet, testSet, valSet = random_split(spect_list_mel, [.7, .2, .1])
print(len(trainSet), len(testSet), len(valSet))