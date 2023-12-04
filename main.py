#from data_conversion import DataConversion
from model import WaveNet
import random
import numpy
import torch
import data_conversion
from torch.utils.data import random_split

#print(WaveNet())
#test1 = DataConversion('./dataset/*.mp3')
#test1.load_data()
#spect_list_mel = test1.data_to_mel()
#inputs_before, inputs_after, outputs = test1.make_inputs_outputs(spect_list_mel)
#print(inputs_before[0].shape)
#print(inputs_after[0].shape)
#print(outputs[0].shape)
#test1.display_mel(spect_list_mel, 20)

#trainSet = [inputs_before[:int(len(inputs_before) * 0.7)],
#            inputs_after[:int(len(inputs_before) * 0.7)],
#            outputs[:int(len(inputs_before) * 0.7)]]
#testSet = [inputs_before[int(len(inputs_before) * 0.7):int(len(inputs_before) * 0.9)],
#           inputs_after[int(len(inputs_before) * 0.7):int(len(inputs_before) * 0.9)],
#            outputs[int(len(inputs_before) * 0.7):int(len(inputs_before) * 0.9)]]
#valSet = [inputs_before[int(len(inputs_before) * 0.9):],
#          inputs_after[int(len(inputs_before) * 0.9):],
#            outputs[int(len(inputs_before) * 0.9):]]

#print(len(trainSet), len(testSet), len(valSet))
#print(trainSet[0][0].shape)

# Training loop (modified from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html). 
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    avgloss = 0
    for batch_num, batch in enumerate(dataloader):
        first, middle, end, first_mel, middle_mel, end_mel = [t.to(device) for t in batch]
        if batch_num == 0: print(first.shape)
        first, middle, end = torch.unsqueeze(first, 1), torch.unsqueeze(middle, 1), torch.unsqueeze(end, 1)
        if batch_num == 0: print(first.shape)
        # Compute prediction error
        pred = model(first, end, middle_mel)
        loss = loss_fn(pred, middle)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avgloss += loss.item()
        if (batch_num + 1) % 20 == 0:
            loss, current = loss.item(), (batch_num + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avgloss /= (batch + 1)
    print(f"average training loss: {avgloss:>7f}")


# These are obviously incorrect values, just having something so it compiles
model = WaveNet(1, 32, 32, data_conversion.n_mels, 4, 4, data_conversion.fill_in * data_conversion.global_sr)
lr = 0.001
batch_size = 64
weight_decay = 0
epochs = 15
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
# Get cpu, gpu or mps device for training (from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = model.to(device)
print(model)

for i in range(epochs):
    for data_batch in ["trainbatch1", "trainbatch2", "trainbatch3", "trainbatch4"]:
        train_dataloader = torch.utils.data.DataLoader(torch.load(data_batch + ".pt"), batch_size=batch_size, shuffle=True)
        train(train_dataloader, model, loss_fn, optimizer)
