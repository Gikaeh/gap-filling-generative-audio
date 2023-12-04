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

#print(WaveNet())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test1 = DataConversion('./dataset/*.mp3')
test1.load_data()
mel_spect_train, mel_spect_test = test1.data_to_mel(device)
# test1.display_mel('full', 20)
# test1.display_mel('cut', 20)

X_train_temp, X_test, y_train_temp, y_test = train_test_split(mel_spect_train, mel_spect_test, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

# print(len(X_test), len(X_train), len(X_val), len(y_test), len(y_train), len(y_val))

model = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert the data to PyTorch DataLoader
scaler = MinMaxScaler()
X_train = [scaler.fit_transform(spec) for spec in X_train]
X_train = np.array(X_train, dtype=np.float32)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
val_full = [scaler.fit_transform(spec) for spec in X_val]
X_test = [scaler.fit_transform(spec) for spec in X_test]
X_test = np.array(X_test, dtype=np.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)


y_train = [scaler.fit_transform(spec) for spec in y_train]
y_train = np.array(y_train, dtype=np.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

val_partial = [scaler.fit_transform(spec) for spec in y_val]
y_test = [scaler.fit_transform(spec) for spec in y_test]
y_test = np.array(y_test, dtype=np.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    print(f'Training Epoch {epoch+1}')
    for full_spectrogram, partial_spectrogram in tqdm(train_loader, leave=False):
        # Forward pass
        outputs = model(full_spectrogram)

        # Compute the loss
        loss = criterion(outputs, partial_spectrogram)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_full_spectrogram, val_partial_spectrogram in tqdm(val_loader, leave=False):

            val_outputs = model(val_full_spectrogram)
            val_loss += criterion(val_outputs, val_partial_spectrogram)

        avg_val_loss = val_loss / len(val_full)

    # Test loop
    with torch.no_grad():
        test_loss = 0.0
        for test_full_spectrogram, test_partial_spectrogram in tqdm(test_loader, leave=False):

            test_outputs = model(test_full_spectrogram)
            test_loss += criterion(test_outputs, test_partial_spectrogram)

        avg_test_loss = test_loss / len(X_test)

    print(f'Validation Loss: {avg_val_loss.item()}, Test Loss: {avg_test_loss.item()}')