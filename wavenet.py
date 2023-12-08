#from data_conversion import DataConversion
from model import WaveNet
import random
import numpy
import torch, auraloss
import data_conversion
from torch.utils.data import random_split
import soundfile as sf
from tqdm import tqdm

# Training loop (modified from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html). 
def train(dataloader, model, loss_fn, optimizer, provide_surrounding_audio = 1):
    size = len(dataloader.dataset)
    model.train()
    avgloss = 0
    for batch_num, batch in tqdm(enumerate(dataloader)):
        first, middle, end, first_mel, middle_mel, end_mel = [t.to(device) for t in batch]
        first, middle, end = torch.unsqueeze(first, 1), torch.unsqueeze(middle, 1), torch.unsqueeze(end, 1)
        
        # Compute prediction error
        pred = model(first * provide_surrounding_audio, end * provide_surrounding_audio, upscale_m(middle_mel))
        loss = loss_fn(pred[:, :, :middle.shape[2]], middle)
        if include_auraloss: loss += auraloss_fn(pred[:, :, :middle.shape[2]], middle)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avgloss += loss.item()
        if (batch_num + 1) % 20 == 0:
            loss, current = loss.item(), (batch_num + 1) * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avgloss /= size
    print(f"average training loss: {avgloss:>7f}")
    return (x[0][0] for x in [first, middle, end, pred])

# Get cpu, gpu or mps device for training (from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
upscale_m = torch.nn.Upsample(size = data_conversion.global_sr * data_conversion.either_side, mode = 'nearest')
model = WaveNet(1, 64, data_conversion.n_mels, 128, 4, int(data_conversion.either_side * data_conversion.global_sr))
model = model.to(device)
model.load_state_dict(torch.load("model.pt", map_location=torch.device(device)))
lr = 0.0002
batch_size = 4
weight_decay = 0
epochs = 15
include_auraloss = False
loss_fn = torch.nn.MSELoss()
auraloss_fn = auraloss.freq.MelSTFTLoss(data_conversion.global_sr, device=device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

print(model)

for i in range(epochs):
    for data_batch in range(1, 5):
        print("Epoch" + str(i) + ", batch " + str(data_batch))
        train_dataloader = torch.utils.data.DataLoader(torch.load("dataset/trainbatch" + str(data_batch) + ".pt"), batch_size=batch_size, shuffle=True)
        first, middle, end, pred = train(train_dataloader, model, loss_fn, optimizer, epochs > 5)
        sf.write("output/wavenet_epoch" + str(i) + "_" + str(data_batch) + "_first.wav", first.detach().cpu(), data_conversion.global_sr)
        sf.write("output/wavenet_epoch" + str(i) + "_" + str(data_batch) + "_middle.wav", middle.detach().cpu(), data_conversion.global_sr)
        sf.write("output/wavenet_epoch" + str(i) + "_" + str(data_batch) + "_end.wav", end.detach().cpu(), data_conversion.global_sr)
        sf.write("output/wavenet_epoch" + str(i) + "_" + str(data_batch) + "_out.wav", pred.detach().cpu(), data_conversion.global_sr)
        sf.write("output/wavenet_epoch" + str(i) + "_" + str(data_batch) + "_out10.wav", (pred * 10).detach().cpu(), data_conversion.global_sr)
    torch.save(model.state_dict(), "output/wavenet_model.pt")
