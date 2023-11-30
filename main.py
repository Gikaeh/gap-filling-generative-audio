from data_conversion import DataConversion
from model import WaveNet
import torch
from torch.utils.data import random_split

#print(WaveNet())
test1 = DataConversion('./dataset/*.mp3')
test1.load_data()
spect_list_mel = test1.data_to_mel()
inputs, outputs = test1.make_inputs_outputs(spect_list_mel)
print(inputs[0].shape)
print(outputs[0].shape)
#test1.display_mel(spect_list_mel, 20)

trainSet = [inputs[:int(len(inputs) * 0.7)],
            outputs[:int(len(inputs) * 0.7)]]
testSet = [inputs[int(len(inputs) * 0.7):int(len(inputs) * 0.9)],
            outputs[int(len(inputs) * 0.7):int(len(inputs) * 0.9)]]
valSet = [inputs[int(len(inputs) * 0.9):],
            outputs[int(len(inputs) * 0.9):]]

print(len(trainSet), len(testSet), len(valSet))
print(trainSet[0][0].shape)

# These are obviously incorrect values, just having something so it compiles
model = WaveNet(1, 32, 32, 4)
print(model)
