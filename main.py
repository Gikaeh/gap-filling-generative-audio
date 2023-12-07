import os
from data_conversion import DataConversion
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import model
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train_losses = []
    val_losses = []
    test_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    test1 = DataConversion('./dataset/*.mp3')
    test1.load_data()
    raw_inp, raw_out, mel_inp, mel_out = test1.data_to_mel(True)
    # test1.display_mel('full', 20)
    # test1.display_mel('cut', 20)

    raw_out_train_temp, raw_out_test, raw_inp_train_temp, raw_inp_test, mel_out_train_temp, mel_out_test, mel_inp_train_temp, mel_inp_test = train_test_split(
        raw_out, raw_inp, mel_out, mel_inp, test_size=0.2, random_state=42)
    raw_out_train, raw_out_val, raw_inp_train, raw_inp_val, mel_out_train, mel_out_val, mel_inp_train, mel_inp_val = train_test_split(
        raw_out_train_temp, raw_inp_train_temp, mel_inp_train_temp, mel_out_train_temp, test_size=0.25, random_state=42)
    # print(len(X_test), len(X_train), len(X_val), len(y_test), len(y_train), len(y_val))
    num_resid_channels = int(input("RC:"))
    model = model.WaveNet(1, num_resid_channels, 128, 256, 4, raw_inp[0].shape[1]).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert the data to PyTorch DataLoader
    scaler = MinMaxScaler(feature_range=(-1,1))
    raw_out_train = [scaler.fit_transform(spec) for spec in raw_out_train]
    raw_out_train = np.array(raw_out_train, dtype=np.float32)
    raw_out_train = torch.tensor(raw_out_train, dtype=torch.float32).to(device)

    raw_out_val = [scaler.transform(spec) for spec in raw_out_val]
    raw_out_val = np.array(raw_out_val, dtype=np.float32)
    raw_out_val = torch.tensor(raw_out_val, dtype=torch.float32).to(device)

    raw_out_test = [scaler.transform(spec) for spec in raw_out_test]
    raw_out_test = np.array(raw_out_test, dtype=np.float32)
    raw_out_test = torch.tensor(raw_out_test, dtype=torch.float32).to(device)

    raw_inp_train = [scaler.fit_transform(spec) for spec in raw_inp_train]
    raw_inp_train = np.array(raw_inp_train, dtype=np.float32)
    raw_inp_train = torch.tensor(raw_inp_train, dtype=torch.float32).to(device)

    raw_inp_val = [scaler.transform(spec) for spec in raw_inp_val]
    raw_inp_val = np.array(raw_inp_val, dtype=np.float32)
    raw_inp_val = torch.tensor(raw_inp_val, dtype=torch.float32).to(device)

    raw_inp_test = [scaler.transform(spec) for spec in raw_inp_test]
    raw_inp_test = np.array(raw_inp_test, dtype=np.float32)
    raw_inp_test = torch.tensor(raw_inp_test, dtype=torch.float32).to(device)

    mel_out_train = [scaler.fit_transform(spec) for spec in mel_out_train]
    mel_out_train = np.array(mel_out_train, dtype=np.float32)
    mel_out_train = torch.tensor(mel_out_train, dtype=torch.float32).to(device)

    mel_out_val = [scaler.transform(spec) for spec in mel_out_val]
    mel_out_val = np.array(mel_out_val, dtype=np.float32)
    mel_out_val = torch.tensor(mel_out_val, dtype=torch.float32).to(device)

    mel_out_test = [scaler.transform(spec) for spec in mel_out_test]
    mel_out_test = np.array(mel_out_test, dtype=np.float32)
    mel_out_test = torch.tensor(mel_out_test, dtype=torch.float32).to(device)

    mel_inp_train = [scaler.fit_transform(spec) for spec in mel_inp_train]
    mel_inp_train = np.array(mel_inp_train, dtype=np.float32)
    mel_inp_train = torch.tensor(mel_inp_train, dtype=torch.float32).to(device)

    mel_inp_val = [scaler.transform(spec) for spec in mel_inp_val]
    mel_inp_val = np.array(mel_inp_val, dtype=np.float32)
    mel_inp_val = torch.tensor(mel_inp_val, dtype=torch.float32).to(device)

    mel_inp_test = [scaler.transform(spec) for spec in mel_inp_test]
    mel_inp_test = np.array(mel_inp_test, dtype=np.float32)
    mel_inp_test = torch.tensor(mel_inp_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(raw_inp_train, raw_out_train, mel_inp_train, mel_out_train)
    val_dataset = TensorDataset(raw_inp_val, raw_out_val, mel_inp_val, mel_out_val)
    test_dataset = TensorDataset(raw_inp_test, raw_out_test, mel_inp_test, mel_out_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch {epoch+1}:')
        train_loss = 0.0
        for raw_inp, raw_out, mel_inp, mel_out in tqdm(train_loader, desc='Training: ', leave=False):
            # Forward pass
            outputs = model(raw_inp.to(device), mel_out.to(device))

            # Compute the loss
            loss = criterion(outputs, raw_out.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            

        avg_train_loss = train_loss / len(train_loader)
        print(epoch + 1, train_loss)
        torch.save(model.state_dict(), 'output/WaveNet.pth')

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for raw_inp, raw_out, mel_inp, mel_out in tqdm(val_loader, desc='Validation: ', leave=False):
                # Forward pass
                outputs = model(raw_inp.to(device), mel_out.to(device))

                # Compute the loss
                loss = criterion(outputs, raw_out.to(device))
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
        print(epoch + 1, val_loss)
            
        if False:
            # Test loop
            with torch.no_grad():
                test_loss = 0.0
                for raw_inp, raw_out, mel_inp, mel_out in tqdm(test_loader, desc='Testing: ', leave=False):
                    # Forward pass
                    outputs = model(raw_inp.to(device), mel_out.to(device))

                    # Compute the loss
                    loss = criterion(outputs, raw_out.to(device))
                    test_loss += loss.item()

                avg_test_loss = test_loss / len(test_loader)
            print(epoch + 1, test_loss)
