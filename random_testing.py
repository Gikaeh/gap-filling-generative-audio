# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
        
#         return x
    
# class DataConversion:
#     def __init__(self, file_dir):
#         self.file_dir = file_dir
#         self.data = glob(file_dir)
#         self.y = []
#         self.sr = []
#         self.mel_full = []
#         self.mel_cut = []

#     def load_data(self):
#         print('Loading Data:')

#         for x in tqdm(range(len(self.data))):
#             yt, srt = lb.load(self.data[x], sr = global_sr)
#             # TODO: Change this to upsample/downsample other sample rates
#             assert srt == global_sr
#             self.y.append(yt)

#     def data_to_mel(self):
#             print('Converting to Mel-Spectrogram:')

#             for x in tqdm(range(len(self.data))):
#                 # Extract a 20-second segment
#                 start_time = np.random.uniform(0, max(0, len(self.y[x]) - 20 * global_sr))
#                 segment = self.y[x][int(start_time):int(start_time) + 20 * global_sr]

#                 # Generate mel-spectrogram for the complete 20 seconds
#                 mel_spect = lb.feature.melspectrogram(y=segment, sr=global_sr)
#                 self.mel_full.append(mel_spect)

#                 # Save a 3-second cut from the middle
#                 cut_start = int((len(mel_spect) / 2) - 1)
#                 cut_end = cut_start + 7 

#                 # Set the 3 seconds in the original mel-spectrogram to zero to create the input
#                 mel_spect[:, cut_start:cut_end] = 0
#                 self.mel_cut.append(mel_spect)

#             return self.mel_full, self.mel_cut
    
from data_conversion import DataConversion
from model import WaveNet, SimpleCNN
from tqdm import tqdm
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# #print(WaveNet())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test1 = DataConversion('./dataset/*.mp3')
# test1.load_data()
# mel_spect_train, mel_spect_test = test1.data_to_mel( )
# # test1.display_mel('full', 20)
# # test1.display_mel('cut', 20)

# X_train_temp, X_test, y_train_temp, y_test = train_test_split(mel_spect_train, mel_spect_test, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

# # print(len(X_test), len(X_train), len(X_val), len(y_test), len(y_train), len(y_val))

# model = SimpleCNN().to(device)

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Convert the data to PyTorch DataLoader
# scaler = MinMaxScaler()
# X_train = [scaler.fit_transform(spec) for spec in X_train]
# X_train = np.array(X_train, dtype=np.float32)
# X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)

# X_val = [scaler.fit_transform(spec) for spec in X_val]
# X_val = np.array(X_val, dtype=np.float32)
# X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)

# X_test = [scaler.fit_transform(spec) for spec in X_test]
# X_test = np.array(X_test, dtype=np.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)


# y_train = [scaler.fit_transform(spec) for spec in y_train]
# y_train = np.array(y_train, dtype=np.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

# y_val = [scaler.fit_transform(spec) for spec in y_val]
# y_val = np.array(y_val, dtype=np.float32)
# y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# y_test = [scaler.fit_transform(spec) for spec in y_test]
# y_test = np.array(y_test, dtype=np.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)
# test_dataset = TensorDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=16)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     print(f'Training Epoch {epoch+1}')
#     for full_spectrogram, partial_spectrogram in tqdm(train_loader, leave=False):
#         # Forward pass
#         outputs = model(full_spectrogram.to(device))

#         # Compute the loss
#         loss = criterion(outputs, partial_spectrogram.to(device))

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Validation loop
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0.0
#         for val_full_spectrogram, val_partial_spectrogram in tqdm(val_loader, leave=False):

#             val_outputs = model(val_full_spectrogram.to(device))
#             val_loss += criterion(val_outputs, val_partial_spectrogram.to(device))

#         avg_val_loss = val_loss / len(val_full_spectrogram)

#     # Test loop
#     with torch.no_grad():
#         test_loss = 0.0
#         for test_full_spectrogram, test_partial_spectrogram in tqdm(test_loader, leave=False):

#             test_outputs = model(test_full_spectrogram.to(device))
#             test_loss += criterion(test_outputs, test_partial_spectrogram.to(device))

#         avg_test_loss = test_loss / len(X_test)

#     print(f'Validation Loss: {avg_val_loss.item()}, Test Loss: {avg_test_loss.item()}')


import matplotlib.pyplot as plt
import librosa

# Assuming you have a DataConversion instance named test1
test1 = DataConversion('./dataset/1036800.low.mp3')
test1.load_data()
mel_spect_full, mel_spect_cut = test1.data_to_mel()

# Display the original mel spectrogram
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title('Original Mel Spectrogram')
librosa.display.specshow(librosa.power_to_db(mel_spect_full[0], ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')

# Display the mel spectrogram after the cut
plt.subplot(2, 1, 2)
plt.title('Mel Spectrogram with Cut')
librosa.display.specshow(librosa.power_to_db(mel_spect_cut[0], ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()
