from data_conversion import DataConversion
from model import WaveNet
import random
import numpy as np
import torch
from torch.utils.data import random_split

#print(WaveNet())
test1 = DataConversion('./dataset/*.mp3')
test1.load_data()
spect_list_mel = test1.data_to_mel()
inputs_before, inputs_after, outputs = test1.make_inputs_outputs(spect_list_mel)
print(inputs_before[0].shape)
print(inputs_after[0].shape)
print(outputs[0].shape)
#test1.display_mel(spect_list_mel, 20)

trainSet = [inputs_before[:int(len(inputs_before) * 0.7)],
            inputs_after[:int(len(inputs_before) * 0.7)],
            outputs[:int(len(inputs_before) * 0.7)]]
testSet = [inputs_before[int(len(inputs_before) * 0.7):int(len(inputs_before) * 0.9)],
           inputs_after[int(len(inputs_before) * 0.7):int(len(inputs_before) * 0.9)],
            outputs[int(len(inputs_before) * 0.7):int(len(inputs_before) * 0.9)]]
valSet = [inputs_before[int(len(inputs_before) * 0.9):],
          inputs_after[int(len(inputs_before) * 0.9):],
            outputs[int(len(inputs_before) * 0.9):]]

print(len(trainSet), len(testSet), len(valSet))
print(trainSet[0][0].shape)

# These are obviously incorrect values, just having something so it compiles
model = WaveNet(1, 32, 32, 4, 2 * 22050)
lr = 0.001
batch_size = 64
weight_decay = 0.02
epochs = 15
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
device = "mps"
model = model.to(device)
print(model)

# Training loop (modified slightly from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html to output training accuracy). 
def train(data, model, loss_fn, optimizer):
    size = len(data[0])
    model.train()
    avgloss = 0
    correct = 0
    random.shuffle(data)
    print(size, batch_size)
    for batch in range(size // batch_size):
        X = torch.unsqueeze(torch.tensor(numpy.array(data[0][batch * batch_size : (batch + 1) * batch_size])),1)
        y = torch.tensor(numpy.array(data[2][batch * batch_size : (batch + 1) * batch_size]))
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        avgloss += loss.item()
        if (batch + 1) % 2 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avgloss /= size // batch_size
    correct /= size
    print(f"average training loss: {avgloss:>7f}")
    print(f"training accuracy: {(100*correct):>0.1f}%")
for i in range(epochs):
    train(trainSet, model, loss_fn, optimizer)
